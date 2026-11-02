/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rolling_shutter_motion.hpp"
#include "core/logger.hpp"
#include <rasterization_config.h>
#include "core/path_utils.hpp"
#include "kernels/rolling_shutter_update.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <format>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>

namespace lfs::training {

    namespace {
        template <class T>
        T* device_alloc(size_t n, const std::vector<T>* init = nullptr) {
            T* p = nullptr;
            if (cudaMalloc(reinterpret_cast<void**>(&p), sizeof(T) * std::max<size_t>(n, 1)) != cudaSuccess)
                return nullptr;
            if (init)
                cudaMemcpy(p, init->data(), sizeof(T) * n, cudaMemcpyHostToDevice);
            else
                cudaMemset(p, 0, sizeof(T) * std::max<size_t>(n, 1));
            return p;
        }

        int frame_number(const std::string& name) {
            static const std::regex re(R"(frame_(\d+))");
            std::smatch m;
            return std::regex_search(name, m, re) ? std::stoi(m[1].str()) : -1;
        }

        std::string trim_png(const std::string& n) {
            return n.size() > 4 && n.ends_with(".png") ? n.substr(0, n.size() - 4) : n;
        }
    } // namespace

    std::expected<std::unique_ptr<RollingShutterMotion>, std::string> RollingShutterMotion::create(
        const RollingShutterSettings& settings, const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras) {
        std::unique_ptr<RollingShutterMotion> rs(new RollingShutterMotion());
        rs->settings_ = settings;
        if (!(settings.readout_ms >= 0.0f && settings.readout_ms <= 100.0f))
            return std::unexpected(std::format("--rs-readout-ms must be within [0, 100] (got {})", settings.readout_ms));
        // learned components
        {
            std::stringstream ss(settings.learn);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                tok.erase(std::remove_if(tok.begin(), tok.end(), [](unsigned char c) { return std::isspace(c); }), tok.end());
                if (tok.empty())
                    continue;
                if (tok == "omega")
                    rs->learn_mask_ |= 0x7u;
                else if (tok == "alpha")
                    rs->learn_mask_ |= 0x38u;
                else if (tok == "vel")
                    rs->learn_mask_ |= 0x1C0u;
                else if (tok == "acc")
                    rs->learn_mask_ |= 0xE00u;
                else if (tok == "readout")
                    rs->learn_readout_ = true;
                else
                    return std::unexpected(std::format("--rs-learn: unknown '{}' (use omega, alpha, vel, acc, readout)", tok));
            }
        }
        // the file
        std::ifstream in(lfs::core::utf8_to_path(settings.path));
        if (!in)
            return std::unexpected(std::format("--rolling-shutter: cannot open '{}'", settings.path));
        std::unordered_map<std::string, std::array<float, 12>> rows;
        std::string line;
        int line_no = 0;
        while (std::getline(in, line)) {
            ++line_no;
            if (line.empty() || line[0] == '#')
                continue;
            std::istringstream ls(line);
            std::string name;
            std::array<float, 12> v{};
            ls >> name;
            for (auto& x : v)
                ls >> x;
            if (!ls || name.empty())
                return std::unexpected(std::format("--rolling-shutter: '{}' line {}: name + 12 numbers expected", settings.path, line_no));
            for (float x : v)
                if (!std::isfinite(x))
                    return std::unexpected(std::format("--rolling-shutter: '{}' line {}: non-finite value", settings.path, line_no));
            rows[trim_png(name)] = v;
        }
        // match the training cameras (by image name, with or without a trailing .png)
        struct Match {
            int uid;
            std::string name;
            int frame;
            std::array<float, 12> v;
        };
        std::vector<Match> matched;
        float fx_sum = 0.0f;
        size_t fisheye = 0;
        for (const auto& cam : cameras) {
            if (!cam)
                continue;
            const auto it = rows.find(trim_png(cam->image_name()));
            if (it == rows.end())
                continue;
            if (cam->camera_model_type() != lfs::core::CameraModelType::FISHEYE || cam->is_undistort_prepared())
                return std::unexpected(std::format("--rolling-shutter needs --native-fisheye on fisheye cameras ('{}' is not)", cam->image_name()));
            ++fisheye;
            auto [fx, fy, cx, cy] = cam->get_intrinsics();
            (void)fy;
            (void)cx;
            (void)cy;
            fx_sum += fx;
            matched.push_back({cam->uid(), cam->image_name(), frame_number(cam->image_name()), it->second});
        }
        if (matched.empty())
            return std::unexpected(std::format("--rolling-shutter: no line of '{}' names a training image ({} lines, {} cameras)",
                                               settings.path, rows.size(), cameras.size()));
        std::sort(matched.begin(), matched.end(), [](const Match& a, const Match& b) {
            return a.frame != b.frame ? a.frame < b.frame : a.name < b.name;
        });
        const size_t N = matched.size();
        std::vector<float> init(N * 12);
        std::vector<int> nb(N * 2, -1);
        for (size_t i = 0; i < N; ++i) {
            rs->row_of_uid_[matched[i].uid] = static_cast<int>(i);
            rs->names_.push_back(matched[i].name);
            std::copy(matched[i].v.begin(), matched[i].v.end(), init.begin() + i * 12);
            // time neighbours for the smoothness term: adjacent training images at most 12 video frames apart
            if (i > 0 && matched[i].frame >= 0 && matched[i - 1].frame >= 0 && matched[i].frame - matched[i - 1].frame <= 12)
                nb[i * 2] = static_cast<int>(i - 1);
            if (i + 1 < N && matched[i].frame >= 0 && matched[i + 1].frame >= 0 && matched[i + 1].frame - matched[i].frame <= 12)
                nb[i * 2 + 1] = static_cast<int>(i + 1);
        }
        rs->init_host_ = init;
        // learning units: one unit moves the image by about one pixel at the readout ends (t = +-tau/2) for omega and
        // alpha, and for vel and acc at a distance of one scene unit
        const float fx = fx_sum / static_cast<float>(N);
        const float h = std::max(settings.readout_ms, 1.0f) * 0.5e-3f;
        for (int k = 0; k < 3; ++k) {
            rs->scale_[k] = 1.0f / (fx * h);
            rs->scale_[3 + k] = 2.0f / (fx * h * h);
            rs->scale_[6 + k] = 1.0f / (fx * h);
            rs->scale_[9 + k] = 2.0f / (fx * h * h);
        }
        const std::vector<float> zeros12(N * 12, 0.0f), zerosN(N, 0.0f), readout = {settings.readout_ms * 1e-3f}, zero3(3, 0.0f);
        rs->motion_ = device_alloc<float>(N * 12, &init);
        rs->init_ = device_alloc<float>(N * 12, &init);
        rs->adam_m_ = device_alloc<float>(N * 12, &zeros12);
        rs->adam_v_ = device_alloc<float>(N * 12, &zeros12);
        rs->steps_ = device_alloc<float>(N, &zerosN);
        rs->readout_ = device_alloc<float>(1, &readout);
        rs->readout_state_ = device_alloc<float>(3, &zero3);
        rs->grad_ = device_alloc<float>(13);
        rs->nb_ = device_alloc<int>(N * 2, &nb);
        if (!rs->motion_ || !rs->init_ || !rs->adam_m_ || !rs->adam_v_ || !rs->steps_ || !rs->readout_ || !rs->readout_state_ ||
            !rs->grad_ || !rs->nb_)
            return std::unexpected("--rolling-shutter: device allocation failed");
        // description
        std::vector<float> spd(N), turn(N);
        for (size_t i = 0; i < N; ++i) {
            const float* v = init.data() + i * 12;
            turn[i] = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) * 57.29578f;
            spd[i] = std::sqrt(v[6] * v[6] + v[7] * v[7] + v[8] * v[8]);
        }
        auto med = [](std::vector<float> x) { std::nth_element(x.begin(), x.begin() + x.size() / 2, x.end()); return x[x.size() / 2]; };
        std::string learned;
        const char* comp[4] = {"omega", "alpha", "vel", "acc"};
        for (int g = 0; g < 4; ++g)
            if ((rs->learn_mask_ >> (3 * g)) & 1u)
                learned += std::string(learned.empty() ? "" : ", ") + comp[g];
        if (rs->learn_readout_)
            learned += std::string(learned.empty() ? "" : ", ") + "readout";
        rs->description_ = std::format(
            "Dynamic rolling shutter: {} of {} training cameras from '{}' (median turn {:.1f} deg/s, speed {:.3f} units/s), readout {} ms, "
            "{} fixed-point re-projections; {}",
            N, cameras.size(), settings.path, med(turn), med(spd), settings.readout_ms, settings.iters,
            learned.empty() ? "motion fixed"
                            : std::format("learning {} from iteration {}{} (lr {} px per camera update, readout lr {} ms/step, prior {}, smooth {})",
                                          learned, settings.learn_start,
                                          (rs->learn_mask_ != 0u && rs->learn_readout_ && settings.motion_start > settings.learn_start)
                                              ? std::format(", staged: readout until {}, then the motion", settings.motion_start)
                                              : (rs->learn_mask_ != 0u && settings.motion_start > settings.learn_start
                                                     ? std::format(", motion from {}", settings.motion_start)
                                                     : std::string{}),
                                          settings.lr, settings.readout_lr_ms, settings.prior, settings.smooth));
        (void)fisheye;
        return rs;
    }

    RollingShutterMotion::~RollingShutterMotion() {
        for (void* p : {static_cast<void*>(motion_), static_cast<void*>(init_), static_cast<void*>(adam_m_), static_cast<void*>(adam_v_),
                        static_cast<void*>(steps_), static_cast<void*>(readout_), static_cast<void*>(readout_state_),
                        static_cast<void*>(grad_), static_cast<void*>(nb_)})
            if (p)
                cudaFree(p);
    }

    int RollingShutterMotion::row_for(const lfs::core::Camera& camera) const {
        const auto it = row_of_uid_.find(camera.uid());
        return it == row_of_uid_.end() ? -1 : it->second;
    }

    fast_lfs::rasterization::FastGSRollingShutter RollingShutterMotion::options_for(const lfs::core::Camera& camera, const int iter,
                                                                                     cudaStream_t stream) {
        fast_lfs::rasterization::FastGSRollingShutter rs{};
        const int row = row_for(camera);
        if (row < 0)
            return rs;
        rs.motion = motion_ + static_cast<size_t>(row) * 12;
        rs.readout = readout_;
        rs.iters = std::max(1, settings_.iters);
        if (motion_learning(iter) || readout_learning(iter)) {
            cudaMemsetAsync(grad_, 0, sizeof(float) * 13, stream);
            rs.grad = grad_;
        }
        return rs;
    }

    bool RollingShutterMotion::motion_learning(const int iter) const {
        return learn_mask_ != 0u && iter >= std::max(settings_.learn_start, settings_.motion_start);
    }

    bool RollingShutterMotion::readout_learning(const int iter) const {
        if (!learn_readout_ || iter < settings_.learn_start)
            return false;
        const bool staged = learn_mask_ != 0u && settings_.motion_start > settings_.learn_start;
        return !staged || iter < settings_.motion_start;
    }

    void RollingShutterMotion::step(const lfs::core::Camera& camera, const int iter, cudaStream_t stream) {
        const bool learn_motion = motion_learning(iter), learn_readout = readout_learning(iter);
        if (!learn_motion && !learn_readout)
            return;
        const int row = row_for(camera);
        if (row < 0)
            return;
        kernels::RollingShutterUpdate u;
        u.row = row;
        u.learn_mask = learn_motion ? learn_mask_ : 0u;
        u.learn_readout = learn_readout;
        u.lr = settings_.lr;
        u.readout_lr_ms = settings_.readout_lr_ms;
        u.prior = settings_.prior;
        u.smooth = settings_.smooth;
        std::copy(std::begin(scale_), std::end(scale_), std::begin(u.scale));
        kernels::rolling_shutter_adam_step(motion_, init_, adam_m_, adam_v_, steps_, readout_, readout_state_, grad_, nb_, u, stream);
    }

    std::string RollingShutterMotion::summary(cudaStream_t stream) {
        const size_t N = names_.size();
        std::vector<float> m(N * 12), st(N);
        float ro = 0.0f;
        cudaStreamSynchronize(stream);
        cudaMemcpy(m.data(), motion_, sizeof(float) * N * 12, cudaMemcpyDeviceToHost);
        cudaMemcpy(st.data(), steps_, sizeof(float) * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(&ro, readout_, sizeof(float), cudaMemcpyDeviceToHost);
        // median |correction| per group, in q units (pixels at the readout ends), over cameras updated at least once
        std::string groups;
        const char* comp[4] = {"omega", "alpha", "vel", "acc"};
        size_t updated = 0;
        for (size_t i = 0; i < N; ++i)
            updated += st[i] > 0.0f;
        for (int g = 0; g < 4; ++g) {
            if (!((learn_mask_ >> (3 * g)) & 1u))
                continue;
            std::vector<float> q;
            for (size_t i = 0; i < N; ++i) {
                if (st[i] <= 0.0f)
                    continue;
                float s2 = 0.0f;
                for (int k = 0; k < 3; ++k) {
                    const size_t j = i * 12 + 3 * g + k;
                    const float d = (m[j] - init_host_[j]) / scale_[3 * g + k];
                    s2 += d * d;
                }
                q.push_back(std::sqrt(s2));
            }
            if (q.empty())
                continue;
            std::sort(q.begin(), q.end());
            groups += std::format(", {} median {:.3f} / p90 {:.3f} px", comp[g], q[q.size() / 2], q[q.size() * 9 / 10]);
        }
        return std::format("rolling shutter: readout {:.3f} ms; {} of {} cameras updated{}", ro * 1000.0f, updated, N, groups);
    }

    bool RollingShutterMotion::save(const std::filesystem::path& file, cudaStream_t stream, std::string& error) {
        const size_t N = names_.size();
        std::vector<float> m(N * 12);
        float ro = 0.0f;
        cudaStreamSynchronize(stream);
        cudaMemcpy(m.data(), motion_, sizeof(float) * N * 12, cudaMemcpyDeviceToHost);
        cudaMemcpy(&ro, readout_, sizeof(float), cudaMemcpyDeviceToHost);
        std::ofstream out(file);
        if (!out) {
            error = std::format("cannot write '{}'", lfs::core::path_to_utf8(file));
            return false;
        }
        out << std::format("# rolling-shutter motion learned by LichtFeld (--rolling-shutter {}; {})\n", settings_.path, description_);
        out << std::format("# readout_ms {:.6f}\n", ro * 1000.0f);
        out << "# name wx wy wz ax ay az vx vy vz Ax Ay Az  (mid-readout camera axes: rad/s, rad/s^2, units/s, units/s^2)\n";
        for (size_t i = 0; i < N; ++i) {
            out << names_[i];
            for (int k = 0; k < 12; ++k)
                out << std::format(" {:.9g}", m[i * 12 + k]);
            out << "\n";
        }
        return static_cast<bool>(out);
    }

} // namespace lfs::training

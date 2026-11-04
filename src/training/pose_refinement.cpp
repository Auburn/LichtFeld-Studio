/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "pose_refinement.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "kernels/pose_refinement_math.hpp"
#include "kernels/pose_refinement_update.hpp"
#include <rasterization_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <fstream>
#include <numeric>
#include <regex>
#include <sstream>

namespace lfs::training {

    namespace {
        constexpr double kRadToDeg = 57.295779513082321;

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

        struct Stats {
            double median = 0, p90 = 0, max = 0;
        };
        Stats stats(std::vector<double> v) {
            Stats s;
            if (v.empty())
                return s;
            std::sort(v.begin(), v.end());
            s.median = v[v.size() / 2];
            s.p90 = v[std::min(v.size() - 1, v.size() * 9 / 10)];
            s.max = v.back();
            return s;
        }

        // Jacobi eigen decomposition of a symmetric 4x4 matrix: the eigenvector of the largest eigenvalue.
        std::array<double, 4> top_eigenvector4(std::array<std::array<double, 4>, 4> a) {
            std::array<std::array<double, 4>, 4> v{};
            for (int i = 0; i < 4; ++i)
                v[i][i] = 1.0;
            for (int sweep = 0; sweep < 60; ++sweep) {
                double off = 0;
                for (int p = 0; p < 4; ++p)
                    for (int q = p + 1; q < 4; ++q)
                        off += a[p][q] * a[p][q];
                if (off < 1e-30)
                    break;
                for (int p = 0; p < 4; ++p)
                    for (int q = p + 1; q < 4; ++q) {
                        if (std::abs(a[p][q]) < 1e-300)
                            continue;
                        const double th = 0.5 * std::atan2(2 * a[p][q], a[q][q] - a[p][p]);
                        const double c = std::cos(th), s = std::sin(th);
                        for (int k = 0; k < 4; ++k) {
                            const double akp = a[k][p], akq = a[k][q];
                            a[k][p] = c * akp - s * akq;
                            a[k][q] = s * akp + c * akq;
                        }
                        for (int k = 0; k < 4; ++k) {
                            const double apk = a[p][k], aqk = a[q][k];
                            a[p][k] = c * apk - s * aqk;
                            a[q][k] = s * apk + c * aqk;
                        }
                        for (int k = 0; k < 4; ++k) {
                            const double vkp = v[k][p], vkq = v[k][q];
                            v[k][p] = c * vkp - s * vkq;
                            v[k][q] = s * vkp + c * vkq;
                        }
                    }
            }
            int best = 0;
            for (int i = 1; i < 4; ++i)
                if (a[i][i] > a[best][best])
                    best = i;
            return {v[0][best], v[1][best], v[2][best], v[3][best]};
        }

        // COLMAP qvec (w, x, y, z) of a rotation matrix (row-major)
        std::array<double, 4> rotmat_to_quat(const double R[9]) {
            const double tr = R[0] + R[4] + R[8];
            double w, x, y, z;
            if (tr > 0) {
                const double s = std::sqrt(tr + 1.0) * 2;
                w = 0.25 * s;
                x = (R[7] - R[5]) / s;
                y = (R[2] - R[6]) / s;
                z = (R[3] - R[1]) / s;
            } else if (R[0] > R[4] && R[0] > R[8]) {
                const double s = std::sqrt(1.0 + R[0] - R[4] - R[8]) * 2;
                w = (R[7] - R[5]) / s;
                x = 0.25 * s;
                y = (R[1] + R[3]) / s;
                z = (R[2] + R[6]) / s;
            } else if (R[4] > R[8]) {
                const double s = std::sqrt(1.0 + R[4] - R[0] - R[8]) * 2;
                w = (R[2] - R[6]) / s;
                x = (R[1] + R[3]) / s;
                y = 0.25 * s;
                z = (R[5] + R[7]) / s;
            } else {
                const double s = std::sqrt(1.0 + R[8] - R[0] - R[4]) * 2;
                w = (R[3] - R[1]) / s;
                x = (R[2] + R[6]) / s;
                y = (R[5] + R[7]) / s;
                z = 0.25 * s;
            }
            const double n = std::sqrt(w * w + x * x + y * y + z * z);
            if (w < 0)
                return {-w / n, -x / n, -y / n, -z / n};
            return {w / n, x / n, y / n, z / n};
        }

        bool copy_text_file(const std::filesystem::path& from, const std::filesystem::path& to) {
            std::error_code ec;
            std::filesystem::copy_file(from, to, std::filesystem::copy_options::overwrite_existing, ec);
            return !ec;
        }
    } // namespace

    std::expected<std::unique_ptr<CameraPoseRefinement>, std::string> CameraPoseRefinement::create(
        const PoseRefinementSettings& settings, const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras) {
        std::unique_ptr<CameraPoseRefinement> pr(new CameraPoseRefinement());
        pr->settings_ = settings;
        if (!(settings.lr_deg >= 0.0f) || !(settings.lr_units >= 0.0f) || !std::isfinite(settings.lr_deg) || !std::isfinite(settings.lr_units))
            return std::unexpected("--pose-lr-deg and --pose-lr-units must be finite and >= 0");
        if (settings.hold_until < settings.start)
            return std::unexpected(std::format("--pose-opt-hold ({}) must be >= --pose-opt-start ({})", settings.hold_until, settings.start));
        pr->half_life_ = settings.half_life > 0.0f
                             ? settings.half_life
                             : std::max(1.0f, static_cast<float>(settings.iterations - settings.hold_until) / 7.0f);
        struct Match {
            int uid;
            std::string name;
            int frame;
            int camera_id;
            std::vector<float> w2c, c;
        };
        std::vector<Match> matched;
        double fx_sum = 0.0;
        for (const auto& cam : cameras) {
            if (!cam)
                continue;
            if (cam->camera_model_type() != lfs::core::CameraModelType::FISHEYE || cam->is_undistort_prepared())
                return std::unexpected(std::format("--pose-opt needs --native-fisheye on fisheye cameras ('{}' is not)", cam->image_name()));
            auto w = cam->world_view_transform().cpu().contiguous().to_vector();
            auto c = cam->cam_position().cpu().contiguous().to_vector();
            if (w.size() != 16 || c.size() != 3)
                return std::unexpected(std::format("--pose-opt: unexpected transform shape for '{}'", cam->image_name()));
            fx_sum += cam->focal_x();
            matched.push_back({cam->uid(), cam->image_name(), frame_number(cam->image_name()), cam->camera_id(), std::move(w), std::move(c)});
        }
        if (matched.empty())
            return std::unexpected("--pose-opt: no training cameras");
        std::sort(matched.begin(), matched.end(), [](const Match& a, const Match& b) {
            return a.frame != b.frame ? a.frame < b.frame : a.name < b.name;
        });
        const size_t N = matched.size();
        pr->focal_px_ = static_cast<float>(fx_sum / static_cast<double>(N));
        std::vector<float> w2c0(N * 16), c0(N * 3);
        std::vector<int> nb(N * 2, -1);
        pr->w2c0_host_.resize(N * 16);
        for (size_t i = 0; i < N; ++i) {
            pr->row_of_uid_[matched[i].uid] = static_cast<int>(i);
            pr->names_.push_back(matched[i].name);
            pr->frames_.push_back(matched[i].frame);
            pr->camera_ids_.push_back(matched[i].camera_id);
            for (int k = 0; k < 16; ++k) {
                w2c0[i * 16 + k] = matched[i].w2c[k];
                pr->w2c0_host_[i * 16 + k] = matched[i].w2c[k];
            }
            for (int k = 0; k < 3; ++k)
                c0[i * 3 + k] = matched[i].c[k];
            if (i > 0 && matched[i].frame >= 0 && matched[i - 1].frame >= 0 && matched[i].frame - matched[i - 1].frame <= 12)
                nb[i * 2] = static_cast<int>(i - 1);
            if (i + 1 < N && matched[i].frame >= 0 && matched[i + 1].frame >= 0 && matched[i + 1].frame - matched[i].frame <= 12)
                nb[i * 2 + 1] = static_cast<int>(i + 1);
        }
        const std::vector<float> zeros6(N * 6, 0.0f), zerosN(N, 0.0f);
        pr->pose_ = device_alloc<float>(N * 6, &zeros6);
        pr->w2c0_ = device_alloc<float>(N * 16, &w2c0);
        pr->c0_ = device_alloc<float>(N * 3, &c0);
        pr->adam_m_ = device_alloc<float>(N * 6, &zeros6);
        pr->adam_v_ = device_alloc<float>(N * 6, &zeros6);
        pr->steps_ = device_alloc<float>(N, &zerosN);
        pr->grad_ = device_alloc<float>(6);
        pr->w2c_out_ = device_alloc<float>(16);
        pr->cam_out_ = device_alloc<float>(3);
        pr->nb_ = device_alloc<int>(N * 2, &nb);
        if (!pr->pose_ || !pr->w2c0_ || !pr->c0_ || !pr->adam_m_ || !pr->adam_v_ || !pr->steps_ || !pr->grad_ || !pr->w2c_out_ ||
            !pr->cam_out_ || !pr->nb_)
            return std::unexpected("--pose-opt: device allocation failed");
        const double visits = static_cast<double>(std::max(0, settings.iterations - settings.start)) / static_cast<double>(N);
        const double end_factor = pr->lr_factor(settings.iterations);
        pr->description_ = std::format(
            "Pose refinement: {} training cameras, lr {} deg and {} units per camera update from iteration {}, held to {}, then "
            "halving every {:.0f} iterations ({:.2g} of the rate at iteration {}); prior {}, smooth {} (pixel units: focal {:.1f} px); "
            "~{:.1f} updates per camera",
            N, settings.lr_deg, settings.lr_units, settings.start, settings.hold_until, pr->half_life_, end_factor,
            settings.iterations, settings.prior, settings.smooth, pr->focal_px_, visits);
        if (end_factor > 0.0101)
            LOG_WARN("--pose-opt: the rate is still {:.3g} of its start at the last iteration; cameras still moving at the end blur "
                     "the splat (lower --pose-opt-half-life or --pose-opt-hold)",
                     end_factor);
        return pr;
    }

    CameraPoseRefinement::~CameraPoseRefinement() {
        for (void* p : {static_cast<void*>(pose_), static_cast<void*>(w2c0_), static_cast<void*>(c0_), static_cast<void*>(adam_m_),
                        static_cast<void*>(adam_v_), static_cast<void*>(steps_), static_cast<void*>(grad_),
                        static_cast<void*>(w2c_out_), static_cast<void*>(cam_out_), static_cast<void*>(nb_)})
            if (p)
                cudaFree(p);
    }

    int CameraPoseRefinement::row_for(const lfs::core::Camera& camera) const {
        const auto it = row_of_uid_.find(camera.uid());
        return it == row_of_uid_.end() ? -1 : it->second;
    }

    float CameraPoseRefinement::lr_factor(const int iter) const {
        if (iter < settings_.start)
            return 0.0f;
        if (iter <= settings_.hold_until)
            return 1.0f;
        return std::exp2(-static_cast<float>(iter - settings_.hold_until) / half_life_);
    }

    fast_lfs::rasterization::FastGSPose CameraPoseRefinement::options_for(const lfs::core::Camera& camera, const int iter,
                                                                          cudaStream_t stream) {
        fast_lfs::rasterization::FastGSPose pose{};
        const int row = row_for(camera);
        if (row < 0 || !active(iter))
            return pose;
        kernels::pose_compose(pose_, w2c0_, c0_, row, w2c_out_, cam_out_, stream);
        pose.w2c = w2c_out_;
        pose.cam_position = cam_out_;
        if (lr_factor(iter) > 0.0f) {
            cudaMemsetAsync(grad_, 0, sizeof(float) * 6, stream);
            pose.grad = grad_;
        }
        return pose;
    }

    void CameraPoseRefinement::step(const lfs::core::Camera& camera, const int iter, cudaStream_t stream) {
        const float f = lr_factor(iter);
        if (f <= 0.0f)
            return;
        const int row = row_for(camera);
        if (row < 0)
            return;
        kernels::PoseRefinementUpdate u;
        u.row = row;
        u.lr_rot = static_cast<float>(settings_.lr_deg / kRadToDeg) * f;
        u.lr_trans = settings_.lr_units * f;
        u.prior = settings_.prior;
        u.smooth = settings_.smooth;
        u.focal_px = focal_px_;
        kernels::pose_adam_step(pose_, adam_m_, adam_v_, steps_, grad_, nb_, w2c0_, u, stream);
    }

    void CameraPoseRefinement::download(std::vector<float>& pose, std::vector<float>& steps, cudaStream_t stream) const {
        const size_t N = names_.size();
        pose.resize(N * 6);
        steps.resize(N);
        cudaStreamSynchronize(stream);
        cudaMemcpy(pose.data(), pose_, sizeof(float) * N * 6, cudaMemcpyDeviceToHost);
        cudaMemcpy(steps.data(), steps_, sizeof(float) * N, cudaMemcpyDeviceToHost);
    }

    std::string CameraPoseRefinement::summary(const int iter, cudaStream_t stream) {
        std::vector<float> p, st;
        download(p, st, stream);
        const size_t N = names_.size();
        if (!settings_.snapshot_dir.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(settings_.snapshot_dir, ec);
            std::ofstream snap(settings_.snapshot_dir / std::format("iter_{:06d}.tsv", iter));
            snap << "name\tupdates\tr_x\tr_y\tr_z\td_x\td_y\td_z\n"; // offsets in the start camera's axes (rad, units)
            for (size_t i = 0; i < N; ++i)
                snap << std::format("{}\t{:.0f}\t{:.9g}\t{:.9g}\t{:.9g}\t{:.9g}\t{:.9g}\t{:.9g}\n", names_[i], st[i], p[i * 6], p[i * 6 + 1],
                                    p[i * 6 + 2], p[i * 6 + 3], p[i * 6 + 4], p[i * 6 + 5]);
        }
        std::vector<double> mv, tn;
        double mean_w[3] = {0, 0, 0}, mean_r[3] = {0, 0, 0}, visits = 0;
        size_t updated = 0;
        for (size_t i = 0; i < N; ++i) {
            visits += st[i];
            if (st[i] <= 0.0f)
                continue;
            ++updated;
            const float* o = p.data() + i * 6;
            mv.push_back(std::sqrt(double(o[3]) * o[3] + double(o[4]) * o[4] + double(o[5]) * o[5]));
            tn.push_back(std::sqrt(double(o[0]) * o[0] + double(o[1]) * o[1] + double(o[2]) * o[2]) * kRadToDeg);
            const double* W = w2c0_host_.data() + i * 16;
            for (int j = 0; j < 3; ++j) {
                mean_w[j] += W[j] * o[3] + W[4 + j] * o[4] + W[8 + j] * o[5];
                mean_r[j] += (W[j] * o[0] + W[4 + j] * o[1] + W[8 + j] * o[2]) * kRadToDeg;
            }
        }
        const Stats m = stats(mv), t = stats(tn);
        const double n = std::max<double>(1.0, static_cast<double>(updated));
        return std::format(
            "pose refinement: rate x{:.3g}; {} of {} cameras updated ({:.1f} updates each on average); move median {:.5f} / p90 {:.5f} / "
            "max {:.5f} units, turn median {:.4f} / p90 {:.4f} / max {:.4f} deg; common move {:.5f} units, common turn {:.4f} deg",
            lr_factor(iter), updated, N, visits / std::max<double>(1.0, static_cast<double>(N)), m.median, m.p90, m.max, t.median, t.p90,
            t.max, std::sqrt(mean_w[0] * mean_w[0] + mean_w[1] * mean_w[1] + mean_w[2] * mean_w[2]) / n,
            std::sqrt(mean_r[0] * mean_r[0] + mean_r[1] * mean_r[1] + mean_r[2] * mean_r[2]) / n);
    }

    bool CameraPoseRefinement::export_results(const std::filesystem::path& out_dir, const std::filesystem::path& dataset_dir,
                                              cudaStream_t stream, std::vector<std::string>& report, std::string& error) {
        std::vector<float> p, st;
        download(p, st, stream);
        const size_t N = names_.size();
        // refined transforms (double) from the start transforms the trainer used
        std::vector<std::array<double, 16>> W1(N);
        std::vector<std::array<double, 3>> C0(N), C1(N);
        std::vector<double> move(N), turn(N);
        for (size_t i = 0; i < N; ++i) {
            const double* W0 = w2c0_host_.data() + i * 16;
            double off[6];
            for (int k = 0; k < 6; ++k)
                off[k] = p[i * 6 + k];
            double c0[3];
            for (int j = 0; j < 3; ++j)
                c0[j] = -(W0[j] * W0[3] + W0[4 + j] * W0[7] + W0[8 + j] * W0[11]);
            double c1[3];
            pose_math::compose<double>(W0, c0, off, W1[i].data(), c1);
            for (int j = 0; j < 3; ++j) {
                C0[i][j] = c0[j];
                C1[i][j] = c1[j];
            }
            move[i] = std::sqrt(off[3] * off[3] + off[4] * off[4] + off[5] * off[5]);
            turn[i] = std::sqrt(off[0] * off[0] + off[1] * off[1] + off[2] * off[2]) * kRadToDeg;
        }
        std::error_code ec;
        const auto cam_dir = out_dir / "cameras_refined";
        std::filesystem::create_directories(cam_dir, ec);
        if (ec) {
            error = std::format("cannot create '{}'", lfs::core::path_to_utf8(cam_dir));
            return false;
        }
        // the dataset's text model
        std::filesystem::path sparse;
        for (const auto& cand : {dataset_dir / "sparse" / "0", dataset_dir / "sparse", dataset_dir})
            if (std::filesystem::exists(cand / "images.txt") || std::filesystem::exists(cand / "cameras.txt")) {
                sparse = cand;
                break;
            }
        std::unordered_map<std::string, int> row_of_name;
        for (size_t i = 0; i < N; ++i)
            row_of_name[names_[i]] = static_cast<int>(i);
        auto pose_fields = [&](int row) {
            const auto& W = W1[row];
            const double R[9] = {W[0], W[1], W[2], W[4], W[5], W[6], W[8], W[9], W[10]};
            const auto q = rotmat_to_quat(R);
            return std::format("{:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f}", q[0], q[1], q[2], q[3], W[3], W[7], W[11]);
        };
        size_t written = 0;
        {
            std::ofstream out(cam_dir / "images.txt");
            if (!out) {
                error = "cannot write cameras_refined/images.txt";
                return false;
            }
            out << std::format("# poses refined by LichtFeld --pose-opt ({} training cameras; {}); same frame as the exported splat\n", N,
                               description_);
            std::vector<bool> done(N, false);
            if (!sparse.empty() && std::filesystem::exists(sparse / "images.txt")) {
                std::ifstream in(sparse / "images.txt");
                std::vector<std::string> lines;
                std::string line;
                while (std::getline(in, line)) {
                    if (!line.empty() && line.back() == '\r')
                        line.pop_back();
                    lines.push_back(line);
                }
                for (size_t k = 0; k < lines.size(); ++k) {
                    const auto& l = lines[k];
                    if (l.empty() || l[0] == '#') {
                        out << l << "\n";
                        continue;
                    }
                    std::istringstream ls(l);
                    std::string id, qw, qx, qy, qz, tx, ty, tz, cid, name;
                    ls >> id >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> cid >> name;
                    const auto it = name.empty() ? row_of_name.end() : row_of_name.find(name);
                    if (it == row_of_name.end()) {
                        out << l << "\n";
                    } else {
                        out << id << " " << pose_fields(it->second) << " " << cid << " " << name << "\n";
                        done[it->second] = true;
                        ++written;
                    }
                    // its POINTS2D line, as it was
                    if (k + 1 < lines.size()) {
                        out << lines[k + 1] << "\n";
                        ++k;
                    }
                }
            }
            // training cameras the file did not list (or no text model): new entries
            int next_id = 1000000;
            for (size_t i = 0; i < N; ++i)
                if (!done[i]) {
                    out << next_id++ << " " << pose_fields(static_cast<int>(i)) << " " << camera_ids_[i] << " " << names_[i] << "\n\n";
                    ++written;
                }
        }
        bool copied_cameras = false, copied_points = false;
        if (!sparse.empty()) {
            if (std::filesystem::exists(sparse / "cameras.txt"))
                copied_cameras = copy_text_file(sparse / "cameras.txt", cam_dir / "cameras.txt");
            if (std::filesystem::exists(sparse / "points3D.txt"))
                copied_points = copy_text_file(sparse / "points3D.txt", cam_dir / "points3D.txt");
        }
        if (!copied_cameras)
            report.push_back(std::format("WARNING: no cameras.txt copied into '{}' (dataset text model not found under '{}')",
                                         lfs::core::path_to_utf8(cam_dir), lfs::core::path_to_utf8(dataset_dir)));
        (void)copied_points;
        // per-camera table: move = the camera centre's displacement (world axes, scene units), turn = the rotation from the
        // start orientation (rotation vector in world axes, degrees); updates = Adam steps the camera got
        {
            std::ofstream tsv(out_dir / "pose_refinement.tsv");
            tsv << "name\tframe\tupdates\tmove_units\tturn_deg\tmove_x\tmove_y\tmove_z\tturn_x_deg\tturn_y_deg\tturn_z_deg\n";
            for (size_t i = 0; i < N; ++i) {
                const double* W0 = w2c0_host_.data() + i * 16;
                double mw[3], rw[3];
                for (int j = 0; j < 3; ++j) {
                    mw[j] = W0[j] * p[i * 6 + 3] + W0[4 + j] * p[i * 6 + 4] + W0[8 + j] * p[i * 6 + 5];
                    rw[j] = (W0[j] * p[i * 6] + W0[4 + j] * p[i * 6 + 1] + W0[8 + j] * p[i * 6 + 2]) * kRadToDeg;
                }
                tsv << std::format("{}\t{}\t{:.0f}\t{:.7f}\t{:.6f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.6f}\t{:.6f}\t{:.6f}\n", names_[i], frames_[i],
                                   st[i], move[i], turn[i], mw[0], mw[1], mw[2], rw[0], rw[1], rw[2]);
            }
        }
        // statistics and the common drift: the similarity (Horn) from the start centres to the refined ones
        const Stats ms = stats(move), ts = stats(turn);
        report.push_back(std::format("pose refinement: {} cameras written to {} ({}); per-camera move median {:.5f} / p90 {:.5f} / max {:.5f} "
                                     "units, turn median {:.4f} / p90 {:.4f} / max {:.4f} deg",
                                     written, lfs::core::path_to_utf8(cam_dir), lfs::core::path_to_utf8(out_dir / "pose_refinement.tsv"),
                                     ms.median, ms.p90, ms.max, ts.median, ts.p90, ts.max));
        double mu0[3] = {0, 0, 0}, mu1[3] = {0, 0, 0};
        for (size_t i = 0; i < N; ++i)
            for (int j = 0; j < 3; ++j) {
                mu0[j] += C0[i][j] / static_cast<double>(N);
                mu1[j] += C1[i][j] / static_cast<double>(N);
            }
        double S[3][3] = {}, ss0 = 0, ss1 = 0;
        for (size_t i = 0; i < N; ++i) {
            double a[3], b[3];
            for (int j = 0; j < 3; ++j) {
                a[j] = C0[i][j] - mu0[j];
                b[j] = C1[i][j] - mu1[j];
                ss0 += a[j] * a[j];
                ss1 += b[j] * b[j];
            }
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    S[r][c] += a[r] * b[c];
        }
        const std::array<std::array<double, 4>, 4> Nm = {{
            {S[0][0] + S[1][1] + S[2][2], S[1][2] - S[2][1], S[2][0] - S[0][2], S[0][1] - S[1][0]},
            {S[1][2] - S[2][1], S[0][0] - S[1][1] - S[2][2], S[0][1] + S[1][0], S[2][0] + S[0][2]},
            {S[2][0] - S[0][2], S[0][1] + S[1][0], -S[0][0] + S[1][1] - S[2][2], S[1][2] + S[2][1]},
            {S[0][1] - S[1][0], S[2][0] + S[0][2], S[1][2] + S[2][1], -S[0][0] - S[1][1] + S[2][2]},
        }};
        auto q = top_eigenvector4(Nm);
        const double qn = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        for (auto& x : q)
            x /= qn;
        const double w = q[0], x = q[1], y = q[2], z = q[3];
        const double R[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                             2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                             2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)};
        const double scale = ss0 > 0 ? std::sqrt(ss1 / ss0) : 1.0;
        const double angle = 2 * std::acos(std::min(1.0, std::abs(w))) * kRadToDeg;
        std::vector<double> resid(N);
        for (size_t i = 0; i < N; ++i) {
            double r2 = 0;
            for (int j = 0; j < 3; ++j) {
                const double a = C0[i][0] - mu0[0], b = C0[i][1] - mu0[1], c = C0[i][2] - mu0[2];
                const double pred = mu1[j] + scale * (R[j * 3] * a + R[j * 3 + 1] * b + R[j * 3 + 2] * c);
                r2 += (C1[i][j] - pred) * (C1[i][j] - pred);
            }
            resid[i] = std::sqrt(r2);
        }
        // the cameras' common turn (mean world-axis rotation offset)
        double mr[3] = {0, 0, 0};
        for (size_t i = 0; i < N; ++i) {
            const double* W0 = w2c0_host_.data() + i * 16;
            for (int j = 0; j < 3; ++j)
                mr[j] += (W0[j] * p[i * 6] + W0[4 + j] * p[i * 6 + 1] + W0[8 + j] * p[i * 6 + 2]) / static_cast<double>(N);
        }
        const Stats rs = stats(resid);
        report.push_back(std::format(
            "pose refinement: common drift of the camera set (similarity start -> refined, NOT removed: the splat moved with it): "
            "centroid shift {:.5f} units ({:+.5f}, {:+.5f}, {:+.5f}), rotation {:.4f} deg, scale {:+.1f} ppm; mean orientation turn "
            "{:.4f} deg; per-camera move left after removing it: median {:.5f} / p90 {:.5f} / max {:.5f} units",
            std::sqrt((mu1[0] - mu0[0]) * (mu1[0] - mu0[0]) + (mu1[1] - mu0[1]) * (mu1[1] - mu0[1]) + (mu1[2] - mu0[2]) * (mu1[2] - mu0[2])),
            mu1[0] - mu0[0], mu1[1] - mu0[1], mu1[2] - mu0[2], angle, (scale - 1.0) * 1e6,
            std::sqrt(mr[0] * mr[0] + mr[1] * mr[1] + mr[2] * mr[2]) * kRadToDeg, rs.median, rs.p90, rs.max));
        {
            std::ofstream sum(cam_dir / "refinement_summary.txt");
            for (const auto& l : report)
                sum << l << "\n";
            sum << description_ << "\n";
        }
        return true;
    }

} // namespace lfs::training

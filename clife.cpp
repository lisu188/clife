/*
 * MIT License
 *
 * Copyright (c) 2019 Andrzej Lis
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "clife.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#endif
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <X11/keysym.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>
#include <sys/ipc.h>
#include <sys/shm.h>


namespace {
    using Cell = LifeBoard::Cell;
    using CellSet = LifeBoard::CellSet;
    using CellBuffer = LifeBoard::CellBuffer;
    using RenderTarget = LifeBoard::RenderTarget;
    using Backend = LifeBoard::Backend;
    using RuleSet = LifeBoard::RuleSet;

    constexpr int kAutoThreadStripeExtent = 512;
    constexpr std::uint64_t kAutoThreadCellsPerWorker = 512ULL * 1024ULL;
    constexpr int kLargeCombinedBitPackedExtent = 4'096;
    constexpr std::uint64_t kLargeCombinedBitPackedArea = 16'000'000ULL;
    constexpr std::size_t kBitPackedHorizontalCacheRows = 6U;
    constexpr int kPatternPixels = 8;
    constexpr int kInitialWindowExtent = 1'000;
    constexpr int kMinimumWindowExtent = 96;
    constexpr int kScrollbarThickness = 16;
    constexpr int kScrollbarMinThumbExtent = 24;
    constexpr int kViewportPanStep = 64;
    constexpr int kRulesPanelPreferredWidth = 224;
    constexpr int kRulesPanelInnerPadding = 12;
    constexpr int kRulesPanelSectionGap = 10;
    constexpr int kRulesFieldHeight = 28;
    constexpr int kRulesButtonHeight = 30;
    constexpr std::array<std::uint8_t, 18> kNextState = {
            0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 0, 0, 0, 0, 0,
    };

    enum class BenchMode {
        Combined,
        Update,
        Render,
    };

    enum class RenderBackend {
        Auto,
        Scalar,
        Avx2,
    };

    struct RuntimeOptions {
        int width = 10'000;
        int height = 10'000;
        int benchmark_frames = 0;
        int benchmark_warmup = 3;
        double benchmark_min_seconds = 0.0;
        int threads = 0;
        float density = 0.05F;
        std::uint64_t seed = 0;
        BenchMode benchmark_mode = BenchMode::Combined;
        Backend backend = Backend::Auto;
        RenderBackend render_backend = RenderBackend::Auto;
    };

    struct PixelPalette {
        std::uint32_t dead = 0;
        std::uint32_t alive = 0x00FFFFFFU;
    };

    struct UiPalette {
        std::uint32_t track = 0;
        std::uint32_t thumb = 0;
        std::uint32_t thumb_border = 0;
        std::uint32_t corner = 0;
        std::uint32_t panel_background = 0;
        std::uint32_t panel_border = 0;
        std::uint32_t input_background = 0;
        std::uint32_t input_border = 0;
        std::uint32_t input_focus = 0;
        std::uint32_t button_background = 0;
        std::uint32_t button_border = 0;
        std::uint32_t text = 0;
    };

    struct Rect {
        int x = 0;
        int y = 0;
        int width = 0;
        int height = 0;

        [[nodiscard]] bool contains(int px, int py) const {
            return (px >= x) && (py >= y) && (px < (x + width)) && (py < (y + height));
        }
    };

    struct alignas(32) PixelLookupTable {
        std::array<std::uint32_t, 256 * kPatternPixels> entries{};
    };

    struct ByteExpandLookupTable {
        std::array<std::uint8_t, 256 * kPatternPixels> entries{};
    };

    struct FreeDeleter {
        void operator()(std::uint8_t *data) const noexcept {
            std::free(data);
        }
    };

    using AlignedBytes = std::unique_ptr<std::uint8_t, FreeDeleter>;

    [[nodiscard]] constexpr std::uint32_t pack_rule_bits(RuleSet rules) noexcept {
        const RuleSet normalized = rules.normalized();
        return static_cast<std::uint32_t>(normalized.birth_mask) |
               (static_cast<std::uint32_t>(normalized.survive_mask) << 9U);
    }

    [[nodiscard]] constexpr RuleSet unpack_rule_bits(std::uint32_t bits) noexcept {
        return {
                static_cast<std::uint16_t>(bits & RuleSet::kMaskBits),
                static_cast<std::uint16_t>((bits >> 9U) & RuleSet::kMaskBits),
        };
    }

    struct CompiledRule {
        RuleSet rule = RuleSet::conway();
        std::array<std::uint8_t, 18> scalar_lut = kNextState;
        bool use_conway_fast_path = true;
    };

    [[nodiscard]] CompiledRule compile_rule(RuleSet rules) {
        const RuleSet normalized = rules.normalized();
        return {
                normalized,
                normalized.scalar_lookup(),
                normalized.is_conway(),
        };
    }

    class RuntimeRules {
    public:
        explicit RuntimeRules(RuleSet initial_rules = RuleSet::conway())
                : _current_bits(pack_rule_bits(initial_rules)),
                  _pending_bits(pack_rule_bits(initial_rules)),
                  _compiled(compile_rule(initial_rules)) {
        }

        void set(RuleSet rules) {
            _pending_bits.store(pack_rule_bits(rules), std::memory_order_release);
            _pending_dirty.store(true, std::memory_order_release);
        }

        [[nodiscard]] RuleSet current() const {
            return unpack_rule_bits(_current_bits.load(std::memory_order_acquire));
        }

        [[nodiscard]] const CompiledRule &compiled() const {
            return _compiled;
        }

        [[nodiscard]] const CompiledRule &apply_pending() {
            if (_pending_dirty.exchange(false, std::memory_order_acq_rel)) {
                const std::uint32_t bits = _pending_bits.load(std::memory_order_acquire);
                _compiled = compile_rule(unpack_rule_bits(bits));
                _current_bits.store(bits, std::memory_order_release);
            }
            return _compiled;
        }

    private:
        std::atomic<std::uint32_t> _current_bits;
        std::atomic<std::uint32_t> _pending_bits;
        std::atomic<bool> _pending_dirty{false};
        CompiledRule _compiled;
    };

    [[nodiscard]] std::uint32_t scale_component(std::uint8_t value, unsigned long mask) {
        if (mask == 0UL) {
            return 0;
        }

        const unsigned shift = static_cast<unsigned>(__builtin_ctzl(mask));
        const std::uint32_t max_value = static_cast<std::uint32_t>(mask >> shift);
        const std::uint32_t scaled = (static_cast<std::uint32_t>(value) * max_value + 127U) / 255U;
        return (scaled << shift) & static_cast<std::uint32_t>(mask);
    }

    [[nodiscard]] PixelPalette make_palette(const Visual *visual) {
        return {
                0U,
                scale_component(255U, visual->red_mask) |
                scale_component(255U, visual->green_mask) |
                scale_component(255U, visual->blue_mask),
        };
    }

    [[nodiscard]] std::uint32_t make_pixel(const Visual *visual, std::uint8_t red, std::uint8_t green, std::uint8_t blue) {
        return scale_component(red, visual->red_mask) |
               scale_component(green, visual->green_mask) |
               scale_component(blue, visual->blue_mask);
    }

    [[nodiscard]] UiPalette make_ui_palette(const Visual *visual) {
        return {
                make_pixel(visual, 48U, 48U, 48U),
                make_pixel(visual, 144U, 144U, 144U),
                make_pixel(visual, 208U, 208U, 208U),
                make_pixel(visual, 32U, 32U, 32U),
                make_pixel(visual, 20U, 20U, 20U),
                make_pixel(visual, 96U, 96U, 96U),
                make_pixel(visual, 12U, 12U, 12U),
                make_pixel(visual, 140U, 140U, 140U),
                make_pixel(visual, 255U, 215U, 64U),
                make_pixel(visual, 56U, 56U, 56U),
                make_pixel(visual, 128U, 128U, 128U),
                make_pixel(visual, 232U, 232U, 232U),
        };
    }

    [[nodiscard]] PixelLookupTable make_pixel_lookup_table(std::uint32_t dead_pixel, std::uint32_t alive_pixel) {
        PixelLookupTable lut;
        for (std::size_t pattern = 0; pattern < 256; ++pattern) {
            auto *entry = lut.entries.data() + pattern * kPatternPixels;
            for (int bit = 0; bit < kPatternPixels; ++bit) {
                entry[bit] = ((pattern >> bit) & 1U) != 0U ? alive_pixel : dead_pixel;
            }
        }
        return lut;
    }

    [[nodiscard]] ByteExpandLookupTable make_byte_expand_lookup_table() {
        ByteExpandLookupTable lut;
        for (std::size_t pattern = 0; pattern < 256; ++pattern) {
            auto *entry = lut.entries.data() + pattern * kPatternPixels;
            for (int bit = 0; bit < kPatternPixels; ++bit) {
                entry[bit] = static_cast<std::uint8_t>((pattern >> bit) & 1U);
            }
        }
        return lut;
    }

    const ByteExpandLookupTable kByteExpandLookupTable = make_byte_expand_lookup_table();

    [[nodiscard]] AlignedBytes allocate_aligned_bytes(std::size_t size, std::size_t alignment) {
        const std::size_t aligned_size = (size + alignment - 1U) & ~(alignment - 1U);
        void *data = nullptr;
        if (posix_memalign(&data, alignment, aligned_size) != 0) {
            return AlignedBytes(nullptr);
        }
        return AlignedBytes(static_cast<std::uint8_t *>(data));
    }

#if defined(__i386__) || defined(__x86_64__)
    [[nodiscard]] bool detect_avx2() {
#if defined(__GNUC__) || defined(__clang__)
        static const bool supported = __builtin_cpu_supports("avx2");
        return supported;
#else
        return false;
#endif
    }
#else
    [[nodiscard]] bool detect_avx2() {
        return false;
    }
#endif

    [[nodiscard]] bool env_to_string(const char *name, std::string &value) {
        const char *raw = std::getenv(name);
        if ((raw == nullptr) || (*raw == '\0')) {
            return false;
        }
        value.assign(raw);
        return true;
    }

    [[nodiscard]] int parse_env_int(const char *name, int fallback, int minimum = 0) {
        const char *raw = std::getenv(name);
        if ((raw == nullptr) || (*raw == '\0')) {
            return fallback;
        }

        char *end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end == raw) {
            return fallback;
        }
        return std::max(minimum, static_cast<int>(parsed));
    }

    [[nodiscard]] float parse_env_float(const char *name, float fallback) {
        const char *raw = std::getenv(name);
        if ((raw == nullptr) || (*raw == '\0')) {
            return fallback;
        }

        char *end = nullptr;
        const float parsed = std::strtof(raw, &end);
        if (end == raw) {
            return fallback;
        }
        return std::clamp(parsed, 0.0F, 1.0F);
    }

    [[nodiscard]] double parse_env_double(const char *name, double fallback, double minimum = 0.0) {
        const char *raw = std::getenv(name);
        if ((raw == nullptr) || (*raw == '\0')) {
            return fallback;
        }

        char *end = nullptr;
        const double parsed = std::strtod(raw, &end);
        if (end == raw) {
            return fallback;
        }
        return std::max(minimum, parsed);
    }

    [[nodiscard]] std::uint64_t parse_env_u64(const char *name, std::uint64_t fallback) {
        const char *raw = std::getenv(name);
        if ((raw == nullptr) || (*raw == '\0')) {
            return fallback;
        }

        char *end = nullptr;
        const unsigned long long parsed = std::strtoull(raw, &end, 10);
        if (end == raw) {
            return fallback;
        }
        return static_cast<std::uint64_t>(parsed);
    }

    [[nodiscard]] Backend parse_backend_from_string(std::string_view value) {
        if ((value == "byte") || (value == "bytes")) {
            return Backend::Byte;
        }
        if ((value == "bitpack") || (value == "bitpacked")) {
            return Backend::BitPacked;
        }
        if ((value == "reference") || (value == "scalar")) {
            return Backend::Reference;
        }
        return Backend::Auto;
    }

    [[nodiscard]] BenchMode parse_bench_mode_from_string(std::string_view value) {
        if (value == "update") {
            return BenchMode::Update;
        }
        if (value == "render") {
            return BenchMode::Render;
        }
        return BenchMode::Combined;
    }

    [[nodiscard]] RenderBackend parse_render_backend_from_string(std::string_view value) {
        if (value == "scalar") {
            return RenderBackend::Scalar;
        }
        if (value == "avx2") {
            return RenderBackend::Avx2;
        }
        return RenderBackend::Auto;
    }

    [[nodiscard]] const char *backend_name_for_enum(Backend backend) {
        switch (backend) {
            case Backend::Byte:
                return "byte";
            case Backend::BitPacked:
                return "bitpacked";
            case Backend::Reference:
                return "reference";
            case Backend::Auto:
            default:
                return "auto";
        }
    }

    [[nodiscard]] const char *bench_mode_name(BenchMode mode) {
        switch (mode) {
            case BenchMode::Update:
                return "update";
            case BenchMode::Render:
                return "render";
            case BenchMode::Combined:
            default:
                return "combined";
        }
    }

    [[nodiscard]] const char *render_backend_name(RenderBackend backend) {
        switch (backend) {
            case RenderBackend::Scalar:
                return "scalar";
            case RenderBackend::Avx2:
                return "avx2";
            case RenderBackend::Auto:
            default:
                return "auto";
        }
    }

    [[nodiscard]] int detect_hardware_threads() {
        return static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
    }

    [[nodiscard]] int ceil_div(int value, int divisor) {
        return (value + divisor - 1) / divisor;
    }

    [[nodiscard]] std::uint64_t ceil_div(std::uint64_t value, std::uint64_t divisor) {
        return (value + divisor - 1ULL) / divisor;
    }

    // Keep auto-threaded work chunks large enough that synchronization does not dominate.
    [[nodiscard]] int select_thread_count(int requested_threads, int width, int height) {
        if (requested_threads > 0) {
            return requested_threads;
        }

        const int safe_width = std::max(1, width);
        const int safe_height = std::max(1, height);
        const int stripe_limited =
                ceil_div(std::max(safe_width, safe_height), kAutoThreadStripeExtent);
        const std::uint64_t area =
                static_cast<std::uint64_t>(safe_width) * static_cast<std::uint64_t>(safe_height);
        const int area_limited = static_cast<int>(ceil_div(area, kAutoThreadCellsPerWorker));
        return std::clamp(std::min(stripe_limited, area_limited), 1, detect_hardware_threads());
    }

    [[nodiscard]] RuntimeOptions load_runtime_options() {
        RuntimeOptions options;
        options.width = parse_env_int("CLIFE_BENCH_WIDTH", options.width, 1);
        options.height = parse_env_int("CLIFE_BENCH_HEIGHT", options.height, 1);
        options.benchmark_frames = parse_env_int("CLIFE_BENCH_FRAMES", 0, 0);
        options.benchmark_warmup = parse_env_int("CLIFE_BENCH_WARMUP", options.benchmark_warmup, 0);
        options.benchmark_min_seconds =
                parse_env_double("CLIFE_BENCH_MIN_SECONDS", options.benchmark_min_seconds, 0.0);
        options.threads = parse_env_int("CLIFE_BENCH_THREADS", options.threads, 0);
        options.density = parse_env_float("CLIFE_BENCH_DENSITY", options.density);
        options.seed = parse_env_u64("CLIFE_BENCH_SEED",
                                     options.benchmark_frames > 0
                                             ? 1ULL
                                             : static_cast<std::uint64_t>(std::time(nullptr)));

        std::string value;
        if (env_to_string("CLIFE_BENCH_BACKEND", value) || env_to_string("CLIFE_BACKEND", value)) {
            options.backend = parse_backend_from_string(value);
        }
        if (env_to_string("CLIFE_BENCH_MODE", value)) {
            options.benchmark_mode = parse_bench_mode_from_string(value);
        }
        if (env_to_string("CLIFE_RENDER_BACKEND", value)) {
            options.render_backend = parse_render_backend_from_string(value);
        }
        return options;
    }

    [[nodiscard]] Backend select_auto_backend(int width, int height, bool update_only) {
        const int safe_width = std::max(1, width);
        const int safe_height = std::max(1, height);
        const int max_extent = std::max(safe_width, safe_height);
        const std::uint64_t area =
                static_cast<std::uint64_t>(safe_width) * static_cast<std::uint64_t>(safe_height);

        if (update_only) {
            return Backend::BitPacked;
        }

        const bool small_combined_board = max_extent <= 1024;
        const bool large_combined_board =
                (area >= kLargeCombinedBitPackedArea) || (max_extent >= kLargeCombinedBitPackedExtent);
        return (small_combined_board || large_combined_board)
                ? Backend::BitPacked
                : Backend::Byte;
    }

    [[nodiscard]] Backend select_runtime_backend(const RuntimeOptions &options) {
        if (options.backend != Backend::Auto) {
            return options.backend;
        }

        return select_auto_backend(
                options.width,
                options.height,
                (options.benchmark_frames > 0) && (options.benchmark_mode == BenchMode::Update));
    }

    [[nodiscard]] int infer_extent(const std::shared_ptr<CellSet> &board, int requested, bool x_axis) {
        if (requested > 0) {
            return requested;
        }

        int inferred = 1;
        if (!board) {
            return inferred;
        }

        for (const Cell &cell: *board) {
            const int value = x_axis ? cell.first : cell.second;
            if (value >= 0) {
                inferred = std::max(inferred, value + 1);
            }
        }
        return inferred;
    }

    [[nodiscard]] int wrap_coordinate(int value, int extent) {
        const int wrapped = value % extent;
        return wrapped >= 0 ? wrapped : wrapped + extent;
    }

    [[nodiscard]] std::uint64_t mix_seed(std::uint64_t seed) {
        seed += 0x9E3779B97F4A7C15ULL;
        seed = (seed ^ (seed >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        seed = (seed ^ (seed >> 27U)) * 0x94D049BB133111EBULL;
        return seed ^ (seed >> 31U);
    }

    class FastRng {
    public:
        explicit FastRng(std::uint64_t seed)
                : _state(seed == 0 ? 0xA5A5A5A5A5A5A5A5ULL : seed) {
        }

        [[nodiscard]] std::uint64_t next() {
            std::uint64_t value = _state;
            value ^= value >> 12U;
            value ^= value << 25U;
            value ^= value >> 27U;
            _state = value;
            return value * 2685821657736338717ULL;
        }

    private:
        std::uint64_t _state;
    };

    [[nodiscard]] CellBuffer make_seed_cells(int width, int height, float density, std::uint64_t seed) {
        const std::size_t total = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        CellBuffer cells(total, 0);

        if (density <= 0.0F) {
            return cells;
        }
        if (density >= 1.0F) {
            std::fill(cells.begin(), cells.end(), static_cast<std::uint8_t>(1));
            return cells;
        }

        const auto threshold = static_cast<std::uint64_t>(
                std::clamp(static_cast<double>(density), 0.0, 1.0) *
                (static_cast<double>(std::numeric_limits<std::uint32_t>::max()) + 1.0));
        FastRng rng(mix_seed(seed) ^
                    (mix_seed(static_cast<std::uint64_t>(width)) << 1U) ^
                    (mix_seed(static_cast<std::uint64_t>(height)) << 2U));
        std::size_t index = 0;
        for (; (index + 1U) < total; index += 2U) {
            const std::uint64_t random_pair = rng.next();
            cells[index] = static_cast<std::uint8_t>(static_cast<std::uint32_t>(random_pair) < threshold);
            cells[index + 1U] =
                    static_cast<std::uint8_t>(static_cast<std::uint32_t>(random_pair >> 32U) < threshold);
        }
        if (index < total) {
            cells[index] = static_cast<std::uint8_t>(static_cast<std::uint32_t>(rng.next()) < threshold);
        }
        return cells;
    }

    struct TaskRange {
        int begin_row = 0;
        int end_row = 0;
    };

    class StaticExecutor {
    public:
        explicit StaticExecutor(std::size_t background_threads) {
            _threads.reserve(background_threads);
            for (std::size_t index = 0; index < background_threads; ++index) {
                _threads.emplace_back(&StaticExecutor::worker_loop, this, index + 1U);
            }
        }

        ~StaticExecutor() {
            if (_threads.empty()) {
                return;
            }

            {
                std::lock_guard<std::mutex> lock(_mutex);
                _stop = true;
                ++_generation;
            }
            _work_cv.notify_all();
            for (std::thread &thread: _threads) {
                thread.join();
            }
        }

        template<typename Task>
        void parallel_for(std::size_t task_count, Task &&task) {
            if (task_count == 0U) {
                return;
            }

            using TaskType = std::decay_t<Task>;
            TaskType task_storage(std::forward<Task>(task));
            if (_threads.empty() || (task_count == 1U)) {
                for (std::size_t task_index = 0; task_index < task_count; ++task_index) {
                    task_storage(task_index);
                }
                return;
            }

            {
                std::lock_guard<std::mutex> lock(_mutex);
                _task_count = task_count;
                _task_context = &task_storage;
                _task_entry = [](void *context, std::size_t task_index) {
                    (*static_cast<TaskType *>(context))(task_index);
                };
                _completed_workers = 0;
                ++_generation;
            }
            _work_cv.notify_all();
            task_storage(0U);

            std::unique_lock<std::mutex> lock(_mutex);
            _done_cv.wait(lock, [this]() {
                return _completed_workers == _threads.size();
            });
            _task_entry = nullptr;
            _task_context = nullptr;
        }

    private:
        void worker_loop(std::size_t task_index) {
            std::size_t seen_generation = 0;
            std::unique_lock<std::mutex> lock(_mutex);
            while (true) {
                _work_cv.wait(lock, [this, &seen_generation]() {
                    return _stop || (_generation != seen_generation);
                });
                if (_stop) {
                    return;
                }

                seen_generation = _generation;
                auto entry = _task_entry;
                void *context = _task_context;
                const std::size_t active_tasks = _task_count;
                lock.unlock();
                if ((entry != nullptr) && (task_index < active_tasks)) {
                    entry(context, task_index);
                }
                lock.lock();

                ++_completed_workers;
                if (_completed_workers == _threads.size()) {
                    _done_cv.notify_one();
                }
            }
        }

        std::vector<std::thread> _threads;
        std::mutex _mutex;
        std::condition_variable _work_cv;
        std::condition_variable _done_cv;
        void (*_task_entry)(void *, std::size_t) = nullptr;
        void *_task_context = nullptr;
        std::size_t _task_count = 0;
        std::size_t _generation = 0;
        std::size_t _completed_workers = 0;
        bool _stop = false;
    };

    void update_halo(CellBuffer &cells, int view_width, int view_height, int stride) {
        for (int row = 0; row < view_height; ++row) {
            const int row_index = (row + 1) * stride;
            cells[static_cast<std::size_t>(row_index)] =
                    cells[static_cast<std::size_t>(row_index + view_width)];
            cells[static_cast<std::size_t>(row_index + view_width + 1)] =
                    cells[static_cast<std::size_t>(row_index + 1)];
        }

        const int first_visible_row = stride;
        const int last_visible_row = view_height * stride;
        std::copy_n(cells.begin() + last_visible_row, stride, cells.begin());
        std::copy_n(cells.begin() + first_visible_row,
                    stride,
                    cells.begin() + static_cast<std::ptrdiff_t>((view_height + 1) * stride));
    }

#if defined(__i386__) || defined(__x86_64__)
    __attribute__((target("avx2"))) void store_pixels_avx2(
            std::uint32_t *pixel_row,
            int x,
            const std::uint32_t *pixel_lut,
            unsigned pattern,
            bool stream_stores) {
        const auto *entry = reinterpret_cast<const __m256i *>(
                pixel_lut + static_cast<std::size_t>(pattern) * kPatternPixels);
        const __m256i colored = _mm256_loadu_si256(entry);
        if (stream_stores) {
            _mm256_stream_si256(reinterpret_cast<__m256i *>(pixel_row + x), colored);
        } else {
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(pixel_row + x), colored);
        }
    }

#endif

    class SimulationBackend {
    public:
        virtual ~SimulationBackend() = default;

        virtual void render_current(const RenderTarget &render_target) = 0;
        virtual void step() = 0;
        virtual void step_and_render(const RenderTarget &render_target) = 0;
        virtual void materialize(CellBuffer &cells) const = 0;
        [[nodiscard]] virtual CellSet snapshot() const = 0;
        [[nodiscard]] virtual bool alive(Cell cell) const = 0;
        virtual void set_rules(RuleSet rules) = 0;
        [[nodiscard]] virtual RuleSet rules() const = 0;
        [[nodiscard]] virtual Backend backend() const = 0;
        [[nodiscard]] virtual int width() const = 0;
        [[nodiscard]] virtual int height() const = 0;
    };

    class BitPackedEngine final : public SimulationBackend {
    public:
        BitPackedEngine(std::shared_ptr<CellSet> board,
                        int threads,
                        int width,
                        int height,
                        RuleSet rules)
                : _view_width(infer_extent(board, width, true)),
                  _view_height(infer_extent(board, height, false)),
                  _word_count(std::max(1, (_view_width + 63) / 64)),
                  _last_word_bits((_view_width - 1) % 64 + 1),
                  _last_word_mask(_last_word_bits == 64 ? ~0ULL : ((1ULL << _last_word_bits) - 1ULL)),
                  _parallelism(std::max(1, std::min(std::max(1, threads), _view_height))),
                  _executor(static_cast<std::size_t>(std::max(0, _parallelism - 1))),
                  _front(static_cast<std::size_t>(_word_count) * static_cast<std::size_t>(_view_height), 0ULL),
                  _back(_front.size(), 0ULL),
                  _rules(rules),
                  _task_ranges(static_cast<std::size_t>(_parallelism)),
                  _horizontal_neighbor_cache(
                          static_cast<std::size_t>(_parallelism) *
                          kBitPackedHorizontalCacheRows *
                          static_cast<std::size_t>(_word_count),
                          0ULL) {
            initialize_ranges();
            if (board) {
                for (const Cell &cell: *board) {
                    set_alive(_front,
                              wrap_coordinate(cell.first, _view_width),
                              wrap_coordinate(cell.second, _view_height));
                }
            }
        }

        BitPackedEngine(CellBuffer cells,
                        int threads,
                        int width,
                        int height,
                        RuleSet rules)
                : _view_width(std::max(1, width)),
                  _view_height(std::max(1, height)),
                  _word_count(std::max(1, (_view_width + 63) / 64)),
                  _last_word_bits((_view_width - 1) % 64 + 1),
                  _last_word_mask(_last_word_bits == 64 ? ~0ULL : ((1ULL << _last_word_bits) - 1ULL)),
                  _parallelism(std::max(1, std::min(std::max(1, threads), _view_height))),
                  _executor(static_cast<std::size_t>(std::max(0, _parallelism - 1))),
                  _front(static_cast<std::size_t>(_word_count) * static_cast<std::size_t>(_view_height), 0ULL),
                  _back(_front.size(), 0ULL),
                  _rules(rules),
                  _task_ranges(static_cast<std::size_t>(_parallelism)),
                  _horizontal_neighbor_cache(
                          static_cast<std::size_t>(_parallelism) *
                          kBitPackedHorizontalCacheRows *
                          static_cast<std::size_t>(_word_count),
                          0ULL) {
            initialize_ranges();
            const std::size_t required =
                    static_cast<std::size_t>(_view_width) * static_cast<std::size_t>(_view_height);
            if (cells.size() < required) {
                cells.resize(required, 0);
            }
            for (int y = 0; y < _view_height; ++y) {
                for (int x = 0; x < _view_width; ++x) {
                    if (cells[static_cast<std::size_t>(y) * static_cast<std::size_t>(_view_width) + x] != 0U) {
                        set_alive(_front, x, y);
                    }
                }
            }
        }

        void render_current(const RenderTarget &render_target) override {
            _executor.parallel_for(static_cast<std::size_t>(_parallelism), [this, &render_target](std::size_t task_index) {
                render_chunk(task_index, _front, render_target);
            });
        }

        void step() override {
            const CompiledRule &rules = _rules.apply_pending();
            _executor.parallel_for(static_cast<std::size_t>(_parallelism), [this, &rules](std::size_t task_index) {
                evaluate_chunk(task_index, rules, nullptr);
            });
            _front.swap(_back);
        }

        void step_and_render(const RenderTarget &render_target) override {
            const CompiledRule &rules = _rules.apply_pending();
            _executor.parallel_for(static_cast<std::size_t>(_parallelism), [this, &rules, &render_target](std::size_t task_index) {
                evaluate_chunk(task_index, rules, &render_target);
            });
            _front.swap(_back);
        }

        void materialize(CellBuffer &cells) const override {
            const int stride = _view_width + 2;
            cells.assign(static_cast<std::size_t>(stride) * static_cast<std::size_t>(_view_height + 2), 0);
            for (int row = 0; row < _view_height; ++row) {
                expand_row(row_ptr(_front, row),
                           cells.data() + static_cast<std::ptrdiff_t>((row + 1) * stride + 1));
            }
            update_halo(cells, _view_width, _view_height, stride);
        }

        [[nodiscard]] CellSet snapshot() const override {
            CellSet board;
            for (int y = 0; y < _view_height; ++y) {
                const std::uint64_t *row = row_ptr(_front, y);
                for (int word_index = 0; word_index < _word_count; ++word_index) {
                    std::uint64_t bits = row[word_index];
                    while (bits != 0ULL) {
                        const int bit_index = __builtin_ctzll(bits);
                        const int x = word_index * 64 + bit_index;
                        if (x < _view_width) {
                            board.emplace(x, y);
                        }
                        bits &= bits - 1ULL;
                    }
                }
            }
            return board;
        }

        [[nodiscard]] bool alive(Cell cell) const override {
            const int wrapped_x = wrap_coordinate(cell.first, _view_width);
            const int wrapped_y = wrap_coordinate(cell.second, _view_height);
            const std::uint64_t *row = row_ptr(_front, wrapped_y);
            return ((row[wrapped_x / 64] >> (wrapped_x % 64)) & 1ULL) != 0ULL;
        }

        void set_rules(RuleSet rules) override {
            _rules.set(rules);
        }

        [[nodiscard]] RuleSet rules() const override {
            return _rules.current();
        }

        [[nodiscard]] Backend backend() const override {
            return Backend::BitPacked;
        }

        [[nodiscard]] int width() const override {
            return _view_width;
        }

        [[nodiscard]] int height() const override {
            return _view_height;
        }

    private:
        void initialize_ranges() {
            const int rows_per_chunk = _view_height / _parallelism;
            const int extra_rows = _view_height % _parallelism;
            for (int task = 0; task < _parallelism; ++task) {
                const int begin_row = (task * rows_per_chunk) + std::min(task, extra_rows);
                const int row_count = rows_per_chunk + (task < extra_rows ? 1 : 0);
                _task_ranges[static_cast<std::size_t>(task)] = {begin_row, begin_row + row_count};
            }
        }

        [[nodiscard]] const std::uint64_t *row_ptr(const std::vector<std::uint64_t> &buffer, int row) const {
            return buffer.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(_word_count);
        }

        [[nodiscard]] std::uint64_t *row_ptr(std::vector<std::uint64_t> &buffer, int row) const {
            return buffer.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(_word_count);
        }

        void set_alive(std::vector<std::uint64_t> &buffer, int x, int y) const {
            std::uint64_t *row = row_ptr(buffer, y);
            row[x / 64] |= 1ULL << (x % 64);
        }

        static inline void add_count_bits(
                std::uint64_t &bit0,
                std::uint64_t &bit1,
                std::uint64_t &bit2,
                std::uint64_t &bit3,
                std::uint64_t value) {
            const std::uint64_t carry0 = bit0 & value;
            bit0 ^= value;
            const std::uint64_t carry1 = bit1 & carry0;
            bit1 ^= carry0;
            const std::uint64_t carry2 = bit2 & carry1;
            bit2 ^= carry1;
            bit3 ^= carry2;
        }

        [[nodiscard]] static inline std::uint64_t count_equals_mask(
                std::uint64_t bit0,
                std::uint64_t bit1,
                std::uint64_t bit2,
                std::uint64_t bit3,
                int count) {
            switch (count) {
                case 0:
                    return (~bit0) & (~bit1) & (~bit2) & (~bit3);
                case 1:
                    return bit0 & (~bit1) & (~bit2) & (~bit3);
                case 2:
                    return (~bit0) & bit1 & (~bit2) & (~bit3);
                case 3:
                    return bit0 & bit1 & (~bit2) & (~bit3);
                case 4:
                    return (~bit0) & (~bit1) & bit2 & (~bit3);
                case 5:
                    return bit0 & (~bit1) & bit2 & (~bit3);
                case 6:
                    return (~bit0) & bit1 & bit2 & (~bit3);
                case 7:
                    return bit0 & bit1 & bit2 & (~bit3);
                case 8:
                    return (~bit0) & (~bit1) & (~bit2) & bit3;
                default:
                    return 0ULL;
            }
        }

        [[nodiscard]] static inline std::uint64_t match_count_masks(
                std::uint64_t bit0,
                std::uint64_t bit1,
                std::uint64_t bit2,
                std::uint64_t bit3,
                std::uint16_t mask) {
            std::uint64_t matches = 0ULL;
            for (int count = 0; count <= 8; ++count) {
                if (((mask >> count) & 1U) != 0U) {
                    matches |= count_equals_mask(bit0, bit1, bit2, bit3, count);
                }
            }
            return matches;
        }

        [[nodiscard]] static inline std::uint64_t evolve_from_neighborhood(
                std::uint64_t a,
                std::uint64_t b,
                std::uint64_t c,
                std::uint64_t d,
                std::uint64_t current_word,
                std::uint64_t e,
                std::uint64_t f,
                std::uint64_t g,
                std::uint64_t h,
                std::uint64_t mask) {
            std::uint64_t bit0 = 0ULL;
            std::uint64_t bit1 = 0ULL;
            std::uint64_t bit2 = 0ULL;
            std::uint64_t bit3 = 0ULL;
            add_count_bits(bit0, bit1, bit2, bit3, a);
            add_count_bits(bit0, bit1, bit2, bit3, b);
            add_count_bits(bit0, bit1, bit2, bit3, c);
            add_count_bits(bit0, bit1, bit2, bit3, d);
            add_count_bits(bit0, bit1, bit2, bit3, e);
            add_count_bits(bit0, bit1, bit2, bit3, f);
            add_count_bits(bit0, bit1, bit2, bit3, g);
            add_count_bits(bit0, bit1, bit2, bit3, h);

            const std::uint64_t equals_three = bit0 & bit1 & ~bit2 & ~bit3;
            const std::uint64_t equals_two = (~bit0) & bit1 & ~bit2 & ~bit3;
            return (equals_three | (current_word & equals_two)) & mask;
        }

        [[nodiscard]] static inline std::uint64_t evolve_from_neighborhood_generic(
                std::uint64_t a,
                std::uint64_t b,
                std::uint64_t c,
                std::uint64_t d,
                std::uint64_t current_word,
                std::uint64_t e,
                std::uint64_t f,
                std::uint64_t g,
                std::uint64_t h,
                std::uint64_t valid_mask,
                const RuleSet &rules) {
            std::uint64_t bit0 = 0ULL;
            std::uint64_t bit1 = 0ULL;
            std::uint64_t bit2 = 0ULL;
            std::uint64_t bit3 = 0ULL;
            add_count_bits(bit0, bit1, bit2, bit3, a);
            add_count_bits(bit0, bit1, bit2, bit3, b);
            add_count_bits(bit0, bit1, bit2, bit3, c);
            add_count_bits(bit0, bit1, bit2, bit3, d);
            add_count_bits(bit0, bit1, bit2, bit3, e);
            add_count_bits(bit0, bit1, bit2, bit3, f);
            add_count_bits(bit0, bit1, bit2, bit3, g);
            add_count_bits(bit0, bit1, bit2, bit3, h);

            const std::uint64_t birth_eq_mask =
                    match_count_masks(bit0, bit1, bit2, bit3, rules.birth_mask);
            const std::uint64_t survive_eq_mask =
                    match_count_masks(bit0, bit1, bit2, bit3, rules.survive_mask);
            return (((~current_word) & birth_eq_mask) | (current_word & survive_eq_mask)) & valid_mask;
        }

        [[nodiscard]] std::uint64_t *task_horizontal_cache(std::size_t task_index, std::size_t slot) {
            const std::size_t stride = kBitPackedHorizontalCacheRows * static_cast<std::size_t>(_word_count);
            return _horizontal_neighbor_cache.data() +
                   task_index * stride +
                   slot * static_cast<std::size_t>(_word_count);
        }

        void compute_horizontal_neighbors(
                const std::uint64_t *row,
                std::uint64_t *left,
                std::uint64_t *right) const {
            const unsigned wrap_bit = static_cast<unsigned>(_last_word_bits - 1);
            if (_word_count == 1) {
                const std::uint64_t word = row[0] & _last_word_mask;
                left[0] = ((word << 1U) | ((word >> wrap_bit) & 1ULL)) & _last_word_mask;
                right[0] = ((word >> 1U) | ((word & 1ULL) << wrap_bit)) & _last_word_mask;
                return;
            }

            const int last_word_index = _word_count - 1;
            const std::uint64_t last_word = row[last_word_index] & _last_word_mask;
            left[0] = (row[0] << 1U) | ((last_word >> wrap_bit) & 1ULL);
            right[0] = (row[0] >> 1U) | ((row[1] & 1ULL) << 63U);

            for (int word_index = 1; word_index < last_word_index; ++word_index) {
                const std::uint64_t word = row[word_index];
                left[word_index] = (word << 1U) | (row[word_index - 1] >> 63U);
                right[word_index] = (word >> 1U) | ((row[word_index + 1] & 1ULL) << 63U);
            }

            left[last_word_index] =
                    ((last_word << 1U) | (row[last_word_index - 1] >> 63U)) & _last_word_mask;
            right[last_word_index] =
                    ((last_word >> 1U) | ((row[0] & 1ULL) << wrap_bit)) & _last_word_mask;
        }

        void render_pattern(
                std::uint32_t *pixel_row,
                int x,
                unsigned pattern,
                int pixel_count,
                const RenderTarget &render_target) const {
#if defined(__i386__) || defined(__x86_64__)
            if ((pixel_count == kPatternPixels) && render_target.use_avx2 && (render_target.pixel_lut != nullptr)) {
                store_pixels_avx2(pixel_row,
                                  x,
                                  render_target.pixel_lut,
                                  pattern,
                                  render_target.stream_stores);
                return;
            }
#endif
            for (int bit = 0; bit < pixel_count; ++bit) {
                pixel_row[x + bit] =
                        ((pattern >> bit) & 1U) != 0U ? render_target.alive_pixel : render_target.dead_pixel;
            }
        }

        void render_full_word(
                std::uint32_t *pixel_row,
                int x,
                std::uint64_t packed,
                const RenderTarget &render_target) const {
#if defined(__i386__) || defined(__x86_64__)
            if (render_target.use_avx2 && (render_target.pixel_lut != nullptr)) {
                store_pixels_avx2(pixel_row, x, render_target.pixel_lut, static_cast<unsigned>(packed & 0xFFU), render_target.stream_stores);
                store_pixels_avx2(pixel_row, x + 8, render_target.pixel_lut, static_cast<unsigned>((packed >> 8U) & 0xFFU), render_target.stream_stores);
                store_pixels_avx2(pixel_row, x + 16, render_target.pixel_lut, static_cast<unsigned>((packed >> 16U) & 0xFFU), render_target.stream_stores);
                store_pixels_avx2(pixel_row, x + 24, render_target.pixel_lut, static_cast<unsigned>((packed >> 24U) & 0xFFU), render_target.stream_stores);
                store_pixels_avx2(pixel_row, x + 32, render_target.pixel_lut, static_cast<unsigned>((packed >> 32U) & 0xFFU), render_target.stream_stores);
                store_pixels_avx2(pixel_row, x + 40, render_target.pixel_lut, static_cast<unsigned>((packed >> 40U) & 0xFFU), render_target.stream_stores);
                store_pixels_avx2(pixel_row, x + 48, render_target.pixel_lut, static_cast<unsigned>((packed >> 48U) & 0xFFU), render_target.stream_stores);
                store_pixels_avx2(pixel_row, x + 56, render_target.pixel_lut, static_cast<unsigned>((packed >> 56U) & 0xFFU), render_target.stream_stores);
                return;
            }
#endif
            render_pattern(pixel_row, x, static_cast<unsigned>(packed & 0xFFU), kPatternPixels, render_target);
            render_pattern(pixel_row, x + 8, static_cast<unsigned>((packed >> 8U) & 0xFFU), kPatternPixels, render_target);
            render_pattern(pixel_row, x + 16, static_cast<unsigned>((packed >> 16U) & 0xFFU), kPatternPixels, render_target);
            render_pattern(pixel_row, x + 24, static_cast<unsigned>((packed >> 24U) & 0xFFU), kPatternPixels, render_target);
            render_pattern(pixel_row, x + 32, static_cast<unsigned>((packed >> 32U) & 0xFFU), kPatternPixels, render_target);
            render_pattern(pixel_row, x + 40, static_cast<unsigned>((packed >> 40U) & 0xFFU), kPatternPixels, render_target);
            render_pattern(pixel_row, x + 48, static_cast<unsigned>((packed >> 48U) & 0xFFU), kPatternPixels, render_target);
            render_pattern(pixel_row, x + 56, static_cast<unsigned>((packed >> 56U) & 0xFFU), kPatternPixels, render_target);
        }

        void render_packed_row(const std::uint64_t *row, int row_index, const RenderTarget &render_target) const {
            auto *pixel_row = reinterpret_cast<std::uint32_t *>(
                    render_target.surface_bytes + static_cast<std::ptrdiff_t>(row_index) * render_target.pitch_bytes);
            const int full_words = _view_width / 64;
            for (int word_index = 0; word_index < full_words; ++word_index) {
                render_full_word(pixel_row, word_index * 64, row[word_index], render_target);
            }

            int remaining = _view_width - full_words * 64;
            if (remaining > 0) {
                std::uint64_t packed = row[full_words] & _last_word_mask;
                int x = full_words * 64;
                while (remaining > 0) {
                    const unsigned pattern = static_cast<unsigned>(packed & 0xFFU);
                    const int pixel_count = std::min(kPatternPixels, remaining);
                    render_pattern(pixel_row, x, pattern, pixel_count, render_target);
                    packed >>= 8U;
                    x += pixel_count;
                    remaining -= pixel_count;
                }
            }
        }

        void render_chunk(
                std::size_t task_index,
                const std::vector<std::uint64_t> &source,
                const RenderTarget &render_target) const {
            const TaskRange range = _task_ranges[task_index];
            if (range.begin_row >= range.end_row) {
                return;
            }
            for (int row = range.begin_row; row < range.end_row; ++row) {
                render_packed_row(row_ptr(source, row), row, render_target);
            }
#if defined(__i386__) || defined(__x86_64__)
            if (render_target.use_avx2 && render_target.stream_stores) {
                _mm_sfence();
            }
#endif
        }

        void evaluate_chunk(std::size_t task_index,
                            const CompiledRule &rules,
                            const RenderTarget *render_target) {
            const TaskRange range = _task_ranges[task_index];
            if (range.begin_row >= range.end_row) {
                return;
            }

            const int last_word_index = _word_count - 1;
            auto *upper_left = task_horizontal_cache(task_index, 0U);
            auto *upper_right = task_horizontal_cache(task_index, 1U);
            auto *current_left = task_horizontal_cache(task_index, 2U);
            auto *current_right = task_horizontal_cache(task_index, 3U);
            auto *lower_left = task_horizontal_cache(task_index, 4U);
            auto *lower_right = task_horizontal_cache(task_index, 5U);

            const auto rotate_three = [](auto *&upper, auto *&current, auto *&lower) {
                auto *previous_upper = upper;
                upper = current;
                current = lower;
                lower = previous_upper;
            };

            const auto *upper = row_ptr(_front, range.begin_row == 0 ? _view_height - 1 : range.begin_row - 1);
            const auto *current = row_ptr(_front, range.begin_row);
            int lower_row_index = range.begin_row + 1 == _view_height ? 0 : range.begin_row + 1;
            const auto *lower = row_ptr(_front, lower_row_index);
            compute_horizontal_neighbors(upper, upper_left, upper_right);
            compute_horizontal_neighbors(current, current_left, current_right);
            compute_horizontal_neighbors(lower, lower_left, lower_right);

            for (int row = range.begin_row; row < range.end_row; ++row) {
                std::uint64_t *next = row_ptr(_back, row);
                if (rules.use_conway_fast_path) {
                    for (int word_index = 0; word_index < last_word_index; ++word_index) {
                        next[word_index] = evolve_from_neighborhood(
                                upper_left[word_index],
                                upper[word_index],
                                upper_right[word_index],
                                current_left[word_index],
                                current[word_index],
                                current_right[word_index],
                                lower_left[word_index],
                                lower[word_index],
                                lower_right[word_index],
                                ~0ULL);
                    }

                    next[last_word_index] = evolve_from_neighborhood(
                            upper_left[last_word_index],
                            upper[last_word_index] & _last_word_mask,
                            upper_right[last_word_index],
                            current_left[last_word_index],
                            current[last_word_index] & _last_word_mask,
                            current_right[last_word_index],
                            lower_left[last_word_index],
                            lower[last_word_index] & _last_word_mask,
                            lower_right[last_word_index],
                            _last_word_mask);
                } else {
                    for (int word_index = 0; word_index < last_word_index; ++word_index) {
                        next[word_index] = evolve_from_neighborhood_generic(
                                upper_left[word_index],
                                upper[word_index],
                                upper_right[word_index],
                                current_left[word_index],
                                current[word_index],
                                current_right[word_index],
                                lower_left[word_index],
                                lower[word_index],
                                lower_right[word_index],
                                ~0ULL,
                                rules.rule);
                    }

                    next[last_word_index] = evolve_from_neighborhood_generic(
                            upper_left[last_word_index],
                            upper[last_word_index] & _last_word_mask,
                            upper_right[last_word_index],
                            current_left[last_word_index],
                            current[last_word_index] & _last_word_mask,
                            current_right[last_word_index],
                            lower_left[last_word_index],
                            lower[last_word_index] & _last_word_mask,
                            lower_right[last_word_index],
                            _last_word_mask,
                            rules.rule);
                }

                if (render_target != nullptr) {
                    render_packed_row(next, row, *render_target);
                }

                if ((row + 1) == range.end_row) {
                    continue;
                }

                upper = current;
                current = lower;
                lower_row_index = lower_row_index + 1 == _view_height ? 0 : lower_row_index + 1;
                lower = row_ptr(_front, lower_row_index);
                rotate_three(upper_left, current_left, lower_left);
                rotate_three(upper_right, current_right, lower_right);
                compute_horizontal_neighbors(lower, lower_left, lower_right);
            }

#if defined(__i386__) || defined(__x86_64__)
            if ((render_target != nullptr) && render_target->use_avx2 && render_target->stream_stores) {
                _mm_sfence();
            }
#endif
        }

        void expand_row(const std::uint64_t *row, std::uint8_t *dst) const {
            const int full_words = _view_width / 64;
            for (int word_index = 0; word_index < full_words; ++word_index) {
                std::uint64_t packed = row[word_index];
                for (int byte_index = 0; byte_index < 8; ++byte_index) {
                    const auto *source = kByteExpandLookupTable.entries.data() +
                                         static_cast<std::size_t>(packed & 0xFFU) * kPatternPixels;
                    std::memcpy(dst + word_index * 64 + byte_index * kPatternPixels, source, kPatternPixels);
                    packed >>= 8U;
                }
            }

            int remaining = _view_width - full_words * 64;
            if (remaining > 0) {
                std::uint64_t packed = row[full_words] & _last_word_mask;
                int x = full_words * 64;
                while (remaining > 0) {
                    const int pixel_count = std::min(kPatternPixels, remaining);
                    const auto *source = kByteExpandLookupTable.entries.data() +
                                         static_cast<std::size_t>(packed & 0xFFU) * kPatternPixels;
                    std::memcpy(dst + x, source, static_cast<std::size_t>(pixel_count));
                    packed >>= 8U;
                    x += pixel_count;
                    remaining -= pixel_count;
                }
            }
        }

        const int _view_width;
        const int _view_height;
        const int _word_count;
        const int _last_word_bits;
        const std::uint64_t _last_word_mask;
        const int _parallelism;
        StaticExecutor _executor;
        std::vector<std::uint64_t> _front;
        std::vector<std::uint64_t> _back;
        RuntimeRules _rules;
        std::vector<TaskRange> _task_ranges;
        std::vector<std::uint64_t> _horizontal_neighbor_cache;
    };

    [[nodiscard]] Backend resolve_backend(Backend requested, int width, int height) {
        if (requested != Backend::Auto) {
            return requested;
        }

        std::string value;
        if (env_to_string("CLIFE_BACKEND", value)) {
            requested = parse_backend_from_string(value);
            if (requested != Backend::Auto) {
                return requested;
            }
        }

        return select_auto_backend(width, height, false);
    }

    class SimulationBoard {
    public:
        SimulationBoard(std::shared_ptr<CellSet> board,
                        int threads,
                        int width,
                        int height,
                        Backend requested,
                        RuleSet rules = RuleSet::conway())
                : _backend(create_backend(std::move(board),
                                          threads,
                                          width,
                                          height,
                                          resolve_backend(requested, width, height),
                                          rules)) {
        }

        SimulationBoard(CellBuffer cells,
                        int threads,
                        int width,
                        int height,
                        Backend requested,
                        RuleSet rules = RuleSet::conway())
                : _backend(create_backend(std::move(cells),
                                          threads,
                                          width,
                                          height,
                                          resolve_backend(requested, width, height),
                                          rules)) {
        }

        void render_current(const RenderTarget &render_target) {
            _backend->render_current(render_target);
        }

        void step() {
            _backend->step();
        }

        void step_and_render(const RenderTarget &render_target) {
            _backend->step_and_render(render_target);
        }

        void materialize(CellBuffer &cells) const {
            _backend->materialize(cells);
        }

        [[nodiscard]] CellSet snapshot() const {
            return _backend->snapshot();
        }

        [[nodiscard]] bool alive(Cell cell) const {
            return _backend->alive(cell);
        }

        void set_rules(RuleSet rules) {
            _backend->set_rules(rules);
        }

        [[nodiscard]] RuleSet rules() const {
            return _backend->rules();
        }

        [[nodiscard]] Backend backend() const {
            return _backend->backend();
        }

        [[nodiscard]] int width() const {
            return _backend->width();
        }

        [[nodiscard]] int height() const {
            return _backend->height();
        }

    private:
        static std::unique_ptr<SimulationBackend> create_backend(
                std::shared_ptr<CellSet> board,
                int threads,
                int width,
                int height,
                Backend requested,
                RuleSet rules) {
            (void) requested;
            return std::make_unique<BitPackedEngine>(std::move(board), threads, width, height, rules);
        }

        static std::unique_ptr<SimulationBackend> create_backend(
                CellBuffer cells,
                int threads,
                int width,
                int height,
                Backend requested,
                RuleSet rules) {
            (void) requested;
            return std::make_unique<BitPackedEngine>(std::move(cells), threads, width, height, rules);
        }

        std::unique_ptr<SimulationBackend> _backend;
    };

    struct HeadlessFrameBuffer {
        AlignedBytes bytes;
        int pitch_bytes = 0;
        PixelPalette palette{};
        bool stream_stores = false;
    };

    class X11FrameBuffer {
    public:
        X11FrameBuffer() = default;

        ~X11FrameBuffer() {
            destroy();
        }

        X11FrameBuffer(const X11FrameBuffer &) = delete;
        X11FrameBuffer &operator=(const X11FrameBuffer &) = delete;

        [[nodiscard]] bool create(int width, int height) {
            _image_width = std::max(width, 1);
            _image_height = std::max(height, 1);
            _display = XOpenDisplay(nullptr);
            if (_display == nullptr) {
                std::cerr << "XOpenDisplay failed" << std::endl;
                return false;
            }

            const int screen = DefaultScreen(_display);
            Visual *visual = DefaultVisual(_display, screen);
            const int depth = DefaultDepth(_display, screen);
            _palette = make_palette(visual);
            _ui_palette = make_ui_palette(visual);

            _window_width = std::clamp(_image_width, kMinimumWindowExtent, kInitialWindowExtent);
            _window_height = std::clamp(_image_height, kMinimumWindowExtent, kInitialWindowExtent);
            update_layout();
            (void) center_viewport();

            _window = XCreateSimpleWindow(_display,
                                          RootWindow(_display, screen),
                                          0,
                                          0,
                                          static_cast<unsigned>(_window_width),
                                          static_cast<unsigned>(_window_height),
                                          0,
                                          0,
                                          0);
            if (_window == 0) {
                std::cerr << "XCreateSimpleWindow failed" << std::endl;
                return false;
            }

            XSizeHints size_hints{};
            size_hints.flags = PSize | PMinSize;
            size_hints.width = _window_width;
            size_hints.height = _window_height;
            size_hints.min_width = kMinimumWindowExtent;
            size_hints.min_height = kMinimumWindowExtent;
            XSetWMNormalHints(_display, _window, &size_hints);

            _gc = XCreateGC(_display, _window, 0, nullptr);
            if (_gc == nullptr) {
                std::cerr << "XCreateGC failed" << std::endl;
                return false;
            }

            _wm_delete = XInternAtom(_display, "WM_DELETE_WINDOW", False);
            XSetWMProtocols(_display, _window, &_wm_delete, 1);
            XSelectInput(_display,
                         _window,
                         ExposureMask |
                         StructureNotifyMask |
                         KeyPressMask |
                         ButtonPressMask |
                         ButtonReleaseMask |
                         ButtonMotionMask);
            XMapWindow(_display, _window);
            XSync(_display, False);

            if (XShmQueryExtension(_display)) {
                _image = XShmCreateImage(_display,
                                         visual,
                                         static_cast<unsigned>(depth),
                                         ZPixmap,
                                         nullptr,
                                         &_shm_info,
                                         _image_width,
                                         _image_height);
                if ((_image != nullptr) && (_image->bits_per_pixel == 32)) {
                    const std::size_t buffer_size =
                            static_cast<std::size_t>(_image->bytes_per_line) *
                            static_cast<std::size_t>(_image->height);
                    _shm_info.shmid = shmget(IPC_PRIVATE, buffer_size, IPC_CREAT | 0600);
                    if (_shm_info.shmid >= 0) {
                        _shm_info.shmaddr = static_cast<char *>(shmat(_shm_info.shmid, nullptr, 0));
                        if (_shm_info.shmaddr != reinterpret_cast<char *>(-1)) {
                            _image->data = _shm_info.shmaddr;
                            _shm_info.readOnly = False;
                            if (XShmAttach(_display, &_shm_info) != 0) {
                                XSync(_display, False);
                                shmctl(_shm_info.shmid, IPC_RMID, nullptr);
                                _use_shm = true;
                            } else {
                                shmdt(_shm_info.shmaddr);
                                _shm_info.shmaddr = nullptr;
                            }
                        }
                    }
                }
                if (!_use_shm && (_image != nullptr)) {
                    _image->data = nullptr;
                    XDestroyImage(_image);
                    _image = nullptr;
                }
            }

            if (_image == nullptr) {
                _image = XCreateImage(_display,
                                      visual,
                                      static_cast<unsigned>(depth),
                                      ZPixmap,
                                      0,
                                      nullptr,
                                      _image_width,
                                      _image_height,
                                      32,
                                      0);
                if ((_image == nullptr) || (_image->bits_per_pixel != 32)) {
                    std::cerr << "XCreateImage failed or returned an unsupported format" << std::endl;
                    return false;
                }

                const std::size_t buffer_size =
                        static_cast<std::size_t>(_image->bytes_per_line) * static_cast<std::size_t>(_image->height);
                AlignedBytes data = allocate_aligned_bytes(buffer_size, 64);
                if (!data) {
                    std::cerr << "Failed to allocate X11 image buffer" << std::endl;
                    return false;
                }
                std::memset(data.get(), 0, buffer_size);
                _image->data = reinterpret_cast<char *>(data.release());
            }

            _pitch_bytes = _image->bytes_per_line;
            _stream_stores =
                    ((static_cast<std::uintptr_t>(_pitch_bytes) |
                      reinterpret_cast<std::uintptr_t>(_image->data)) & 31U) == 0U;
            return true;
        }

        void destroy() {
            if (_display == nullptr) {
                return;
            }

            if (_use_shm && (_image != nullptr)) {
                XShmDetach(_display, &_shm_info);
                if (_shm_info.shmaddr != nullptr) {
                    shmdt(_shm_info.shmaddr);
                    _shm_info.shmaddr = nullptr;
                }
                _image->data = nullptr;
            }
            if (_image != nullptr) {
                XDestroyImage(_image);
                _image = nullptr;
            }
            if (_gc != nullptr) {
                XFreeGC(_display, _gc);
                _gc = nullptr;
            }
            if (_window != 0) {
                XDestroyWindow(_display, _window);
                _window = 0;
            }
            XCloseDisplay(_display);
            _display = nullptr;
        }

        [[nodiscard]] bool poll_events(bool &running) {
            bool redraw_requested = false;
            while ((_display != nullptr) && (XPending(_display) > 0)) {
                XEvent event;
                XNextEvent(_display, &event);
                if ((event.type == ClientMessage) &&
                    (static_cast<Atom>(event.xclient.data.l[0]) == _wm_delete)) {
                    running = false;
                } else if (event.type == DestroyNotify) {
                    running = false;
                } else if (event.type == Expose) {
                    redraw_requested = true;
                } else if (event.type == ConfigureNotify) {
                    redraw_requested = handle_configure_notify(event.xconfigure) || redraw_requested;
                } else if (event.type == KeyPress) {
                    redraw_requested = handle_key_press(event.xkey) || redraw_requested;
                } else if (event.type == ButtonPress) {
                    redraw_requested = handle_button_press(event.xbutton) || redraw_requested;
                } else if (event.type == ButtonRelease) {
                    redraw_requested = handle_button_release(event.xbutton) || redraw_requested;
                } else if (event.type == MotionNotify) {
                    redraw_requested = handle_motion_notify(event.xmotion) || redraw_requested;
                }
            }
            return redraw_requested;
        }

        void present() {
            XSetForeground(_display, _gc, _palette.dead);
            XFillRectangle(_display,
                           _window,
                           _gc,
                           _content_rect.x,
                           _content_rect.y,
                           static_cast<unsigned>(_content_rect.width),
                           static_cast<unsigned>(_content_rect.height));

            const int copy_width = std::min(_content_rect.width, std::max(0, _image_width - _viewport_x));
            const int copy_height = std::min(_content_rect.height, std::max(0, _image_height - _viewport_y));
            if ((copy_width > 0) && (copy_height > 0) && (_image != nullptr)) {
                if (_use_shm) {
                    XShmPutImage(_display,
                                 _window,
                                 _gc,
                                 _image,
                                 _viewport_x,
                                 _viewport_y,
                                 _content_rect.x,
                                 _content_rect.y,
                                 static_cast<unsigned>(copy_width),
                                 static_cast<unsigned>(copy_height),
                                 False);
                } else {
                    XPutImage(_display,
                              _window,
                              _gc,
                              _image,
                              _viewport_x,
                              _viewport_y,
                              _content_rect.x,
                              _content_rect.y,
                              static_cast<unsigned>(copy_width),
                              static_cast<unsigned>(copy_height));
                }
            }

            draw_scrollbar(_horizontal_scrollbar_rect, _horizontal_thumb_rect);
            draw_scrollbar(_vertical_scrollbar_rect, _vertical_thumb_rect);
            if ((_corner_rect.width > 0) && (_corner_rect.height > 0)) {
                XSetForeground(_display, _gc, _ui_palette.corner);
                XFillRectangle(_display,
                               _window,
                               _gc,
                               _corner_rect.x,
                               _corner_rect.y,
                               static_cast<unsigned>(_corner_rect.width),
                               static_cast<unsigned>(_corner_rect.height));
            }

            draw_rules_panel();

            XFlush(_display);
        }

        [[nodiscard]] std::uint8_t *pixels() const {
            return reinterpret_cast<std::uint8_t *>(_image->data);
        }

        [[nodiscard]] int pitch_bytes() const {
            return _pitch_bytes;
        }

        [[nodiscard]] const PixelPalette &palette() const {
            return _palette;
        }

        [[nodiscard]] bool stream_stores() const {
            return _stream_stores;
        }

        [[nodiscard]] bool set_current_rules(RuleSet rules) {
            const std::string next_label = rules.format();
            if (next_label == _current_rule_label) {
                return false;
            }
            _current_rule_label = next_label;
            return true;
        }

        [[nodiscard]] bool consume_rule_change(RuleSet &rules) {
            if (!_has_pending_rule_change) {
                return false;
            }
            rules = _pending_rules;
            _has_pending_rule_change = false;
            return true;
        }

    private:
        enum class DragMode {
            Idle,
            HorizontalScrollbar,
            VerticalScrollbar,
            MiddlePan,
        };

        enum class FocusField {
            Inactive,
            Birth,
            Survival,
        };

        [[nodiscard]] static int compute_thumb_extent(int track_extent, int viewport_extent, int image_extent) {
            if (track_extent <= 0) {
                return 0;
            }
            if ((image_extent <= 0) || (viewport_extent >= image_extent)) {
                return track_extent;
            }
            const int minimum_thumb = std::min(kScrollbarMinThumbExtent, track_extent);
            const int scaled_thumb = static_cast<int>(
                    (static_cast<long long>(track_extent) * viewport_extent) / image_extent);
            return std::clamp(scaled_thumb, minimum_thumb, track_extent);
        }

        [[nodiscard]] static int compute_thumb_offset(int viewport_offset,
                                                      int max_viewport_offset,
                                                      int track_extent,
                                                      int thumb_extent) {
            if ((max_viewport_offset <= 0) || (track_extent <= thumb_extent)) {
                return 0;
            }
            const int thumb_travel = track_extent - thumb_extent;
            return static_cast<int>(
                    (static_cast<long long>(viewport_offset) * thumb_travel + max_viewport_offset / 2LL) /
                    max_viewport_offset);
        }

        [[nodiscard]] static int compute_viewport_offset_from_thumb(int thumb_offset,
                                                                    int max_viewport_offset,
                                                                    int track_extent,
                                                                    int thumb_extent) {
            if ((max_viewport_offset <= 0) || (track_extent <= thumb_extent)) {
                return 0;
            }
            const int thumb_travel = track_extent - thumb_extent;
            const int clamped_thumb_offset = std::clamp(thumb_offset, 0, thumb_travel);
            return static_cast<int>(
                    (static_cast<long long>(clamped_thumb_offset) * max_viewport_offset + thumb_travel / 2LL) /
                    thumb_travel);
        }

        [[nodiscard]] static bool append_unique_digit(std::string &digits, char digit) {
            if ((digit < '0') || (digit > '8')) {
                return false;
            }
            if (digits.find(digit) != std::string::npos) {
                return false;
            }
            digits.push_back(digit);
            std::sort(digits.begin(), digits.end());
            return true;
        }

        [[nodiscard]] bool set_focus(FocusField focus) {
            if (_focused_field == focus) {
                return false;
            }
            _focused_field = focus;
            return true;
        }

        [[nodiscard]] std::string *focused_input() {
            switch (_focused_field) {
                case FocusField::Birth:
                    return &_birth_input;
                case FocusField::Survival:
                    return &_survival_input;
                case FocusField::Inactive:
                default:
                    return nullptr;
            }
        }

        [[nodiscard]] bool queue_rule_change(RuleSet rules) {
            _pending_rules = rules.normalized();
            _has_pending_rule_change = true;
            return true;
        }

        [[nodiscard]] bool queue_apply_from_inputs() {
            _birth_input = RuleSet::normalize_digits(_birth_input);
            _survival_input = RuleSet::normalize_digits(_survival_input);
            return queue_rule_change(RuleSet::from_digit_strings(_birth_input, _survival_input));
        }

        [[nodiscard]] bool reset_to_default_rules() {
            const RuleSet defaults = RuleSet::conway();
            _birth_input = defaults.birth_digits();
            _survival_input = defaults.survive_digits();
            return queue_rule_change(defaults);
        }

        void update_panel_layout() {
            const int panel_width = std::min(kRulesPanelPreferredWidth, std::max(0, _window_width - 1));
            const int workspace_width = std::max(1, _window_width - panel_width);
            _panel_rect = {workspace_width, 0, panel_width, _window_height};

            if (_panel_rect.width <= 0) {
                _birth_field_rect = {};
                _survival_field_rect = {};
                _apply_button_rect = {};
                _reset_button_rect = {};
                return;
            }

            const int inner_x = _panel_rect.x + kRulesPanelInnerPadding;
            const int inner_width = std::max(1, _panel_rect.width - 2 * kRulesPanelInnerPadding);
            int y = _panel_rect.y + kRulesPanelInnerPadding + 18;
            y += kRulesPanelSectionGap;
            _birth_field_rect = {inner_x, y + 14, inner_width, kRulesFieldHeight};
            y = _birth_field_rect.y + _birth_field_rect.height + kRulesPanelSectionGap + 14;
            _survival_field_rect = {inner_x, y, inner_width, kRulesFieldHeight};
            y = _survival_field_rect.y + _survival_field_rect.height + kRulesPanelSectionGap + 18;
            _apply_button_rect = {inner_x, y, inner_width, kRulesButtonHeight};
            y = _apply_button_rect.y + _apply_button_rect.height + kRulesPanelSectionGap;
            _reset_button_rect = {inner_x, y, inner_width, kRulesButtonHeight};
        }

        void draw_text(int x, int baseline_y, std::string_view text) {
            if (text.empty()) {
                return;
            }
            XDrawString(_display,
                        _window,
                        _gc,
                        x,
                        baseline_y,
                        text.data(),
                        static_cast<int>(text.size()));
        }

        void draw_input_field(const Rect &rect, std::string_view text, bool focused) {
            if ((rect.width <= 0) || (rect.height <= 0)) {
                return;
            }
            XSetForeground(_display, _gc, _ui_palette.input_background);
            XFillRectangle(_display,
                           _window,
                           _gc,
                           rect.x,
                           rect.y,
                           static_cast<unsigned>(rect.width),
                           static_cast<unsigned>(rect.height));
            XSetForeground(_display, _gc, focused ? _ui_palette.input_focus : _ui_palette.input_border);
            XDrawRectangle(_display,
                           _window,
                           _gc,
                           rect.x,
                           rect.y,
                           static_cast<unsigned>(std::max(0, rect.width - 1)),
                           static_cast<unsigned>(std::max(0, rect.height - 1)));
            XSetForeground(_display, _gc, _ui_palette.text);
            draw_text(rect.x + 8, rect.y + rect.height / 2 + 5, text);
        }

        void draw_button(const Rect &rect, std::string_view label) {
            if ((rect.width <= 0) || (rect.height <= 0)) {
                return;
            }
            XSetForeground(_display, _gc, _ui_palette.button_background);
            XFillRectangle(_display,
                           _window,
                           _gc,
                           rect.x,
                           rect.y,
                           static_cast<unsigned>(rect.width),
                           static_cast<unsigned>(rect.height));
            XSetForeground(_display, _gc, _ui_palette.button_border);
            XDrawRectangle(_display,
                           _window,
                           _gc,
                           rect.x,
                           rect.y,
                           static_cast<unsigned>(std::max(0, rect.width - 1)),
                           static_cast<unsigned>(std::max(0, rect.height - 1)));
            XSetForeground(_display, _gc, _ui_palette.text);
            draw_text(rect.x + 8, rect.y + rect.height / 2 + 5, label);
        }

        void draw_rules_panel() {
            if ((_panel_rect.width <= 0) || (_panel_rect.height <= 0)) {
                return;
            }

            XSetForeground(_display, _gc, _ui_palette.panel_background);
            XFillRectangle(_display,
                           _window,
                           _gc,
                           _panel_rect.x,
                           _panel_rect.y,
                           static_cast<unsigned>(_panel_rect.width),
                           static_cast<unsigned>(_panel_rect.height));
            XSetForeground(_display, _gc, _ui_palette.panel_border);
            XDrawLine(_display,
                      _window,
                      _gc,
                      _panel_rect.x,
                      _panel_rect.y,
                      _panel_rect.x,
                      _panel_rect.y + _panel_rect.height);

            XSetForeground(_display, _gc, _ui_palette.text);
            const int text_x = _panel_rect.x + kRulesPanelInnerPadding;
            int baseline = _panel_rect.y + kRulesPanelInnerPadding + 12;
            draw_text(text_x, baseline, "Rules");
            baseline += 18;
            draw_text(text_x, baseline, std::string("Current: ") + _current_rule_label);

            if (_birth_field_rect.width > 0) {
                draw_text(text_x, _birth_field_rect.y - 6, "Birth");
                draw_input_field(_birth_field_rect, _birth_input, _focused_field == FocusField::Birth);
            }
            if (_survival_field_rect.width > 0) {
                draw_text(text_x, _survival_field_rect.y - 6, "Survival");
                draw_input_field(_survival_field_rect,
                                 _survival_input,
                                 _focused_field == FocusField::Survival);
            }

            draw_button(_apply_button_rect, "Apply");
            draw_button(_reset_button_rect, "Reset To Default");
        }

        void update_layout() {
            bool show_horizontal = false;
            bool show_vertical = false;
            update_panel_layout();
            const int workspace_width = std::max(1, _window_width - _panel_rect.width);
            while (true) {
                const int content_width = std::max(1, workspace_width - (show_vertical ? kScrollbarThickness : 0));
                const int content_height = std::max(1, _window_height - (show_horizontal ? kScrollbarThickness : 0));
                const bool new_show_horizontal = _image_width > content_width;
                const bool new_show_vertical = _image_height > content_height;
                if ((new_show_horizontal == show_horizontal) && (new_show_vertical == show_vertical)) {
                    _content_rect = {0, 0, content_width, content_height};
                    _show_horizontal_scrollbar = show_horizontal;
                    _show_vertical_scrollbar = show_vertical;
                    break;
                }
                show_horizontal = new_show_horizontal;
                show_vertical = new_show_vertical;
            }

            _horizontal_scrollbar_rect = {};
            _horizontal_thumb_rect = {};
            _vertical_scrollbar_rect = {};
            _vertical_thumb_rect = {};
            _corner_rect = {};

            if (_show_horizontal_scrollbar) {
                _horizontal_scrollbar_rect = {
                        0,
                        _content_rect.height,
                        _content_rect.width,
                        kScrollbarThickness,
                };
            }
            if (_show_vertical_scrollbar) {
                _vertical_scrollbar_rect = {
                        _content_rect.width,
                        0,
                        kScrollbarThickness,
                        _content_rect.height,
                };
            }
            if (_show_horizontal_scrollbar && _show_vertical_scrollbar) {
                _corner_rect = {
                        _content_rect.width,
                        _content_rect.height,
                        kScrollbarThickness,
                        kScrollbarThickness,
                };
            }

            (void) set_viewport(_viewport_x, _viewport_y);
        }

        void update_thumb_rects() {
            if (_show_horizontal_scrollbar) {
                const int thumb_width = compute_thumb_extent(
                        _horizontal_scrollbar_rect.width,
                        _content_rect.width,
                        _image_width);
                const int thumb_offset = compute_thumb_offset(
                        _viewport_x,
                        max_viewport_x(),
                        _horizontal_scrollbar_rect.width,
                        thumb_width);
                _horizontal_thumb_rect = {
                        _horizontal_scrollbar_rect.x + thumb_offset,
                        _horizontal_scrollbar_rect.y,
                        thumb_width,
                        _horizontal_scrollbar_rect.height,
                };
            }
            if (_show_vertical_scrollbar) {
                const int thumb_height = compute_thumb_extent(
                        _vertical_scrollbar_rect.height,
                        _content_rect.height,
                        _image_height);
                const int thumb_offset = compute_thumb_offset(
                        _viewport_y,
                        max_viewport_y(),
                        _vertical_scrollbar_rect.height,
                        thumb_height);
                _vertical_thumb_rect = {
                        _vertical_scrollbar_rect.x,
                        _vertical_scrollbar_rect.y + thumb_offset,
                        _vertical_scrollbar_rect.width,
                        thumb_height,
                };
            }
        }

        void draw_scrollbar(const Rect &track, const Rect &thumb) {
            if ((track.width <= 0) || (track.height <= 0)) {
                return;
            }

            XSetForeground(_display, _gc, _ui_palette.track);
            XFillRectangle(_display,
                           _window,
                           _gc,
                           track.x,
                           track.y,
                           static_cast<unsigned>(track.width),
                           static_cast<unsigned>(track.height));

            if ((thumb.width <= 0) || (thumb.height <= 0)) {
                return;
            }

            XSetForeground(_display, _gc, _ui_palette.thumb);
            XFillRectangle(_display,
                           _window,
                           _gc,
                           thumb.x,
                           thumb.y,
                           static_cast<unsigned>(thumb.width),
                           static_cast<unsigned>(thumb.height));
            if ((thumb.width > 1) && (thumb.height > 1)) {
                XSetForeground(_display, _gc, _ui_palette.thumb_border);
                XDrawRectangle(_display,
                               _window,
                               _gc,
                               thumb.x,
                               thumb.y,
                               static_cast<unsigned>(thumb.width - 1),
                               static_cast<unsigned>(thumb.height - 1));
            }
        }

        [[nodiscard]] int max_viewport_x() const {
            return std::max(0, _image_width - _content_rect.width);
        }

        [[nodiscard]] int max_viewport_y() const {
            return std::max(0, _image_height - _content_rect.height);
        }

        [[nodiscard]] bool set_viewport(int x, int y) {
            const int clamped_x = std::clamp(x, 0, max_viewport_x());
            const int clamped_y = std::clamp(y, 0, max_viewport_y());
            if ((clamped_x == _viewport_x) && (clamped_y == _viewport_y)) {
                update_thumb_rects();
                return false;
            }
            _viewport_x = clamped_x;
            _viewport_y = clamped_y;
            update_thumb_rects();
            return true;
        }

        [[nodiscard]] bool pan_viewport(int dx, int dy) {
            return set_viewport(_viewport_x + dx, _viewport_y + dy);
        }

        [[nodiscard]] bool page_viewport(int dx, int dy) {
            const int page_x = std::max(1, _content_rect.width);
            const int page_y = std::max(1, _content_rect.height);
            return set_viewport(_viewport_x + dx * page_x, _viewport_y + dy * page_y);
        }

        [[nodiscard]] bool center_viewport() {
            return set_viewport(max_viewport_x() / 2, max_viewport_y() / 2);
        }

        [[nodiscard]] bool handle_configure_notify(const XConfigureEvent &event) {
            if ((event.width == _window_width) && (event.height == _window_height)) {
                return false;
            }
            _window_width = std::max(1, event.width);
            _window_height = std::max(1, event.height);
            update_layout();
            return true;
        }

        [[nodiscard]] bool handle_key_press(XKeyEvent &event) {
            KeySym keysym = 0;
            char text[8] = {};
            const int text_length = XLookupString(&event, text, static_cast<int>(sizeof(text)), &keysym, nullptr);

            if (_focused_field != FocusField::Inactive) {
                if ((keysym == XK_Return) || (keysym == XK_KP_Enter)) {
                    return queue_apply_from_inputs();
                }
                if (keysym == XK_BackSpace) {
                    std::string *input = focused_input();
                    if ((input != nullptr) && !input->empty()) {
                        input->pop_back();
                        return true;
                    }
                    return false;
                }
                if (keysym == XK_Tab) {
                    return set_focus(_focused_field == FocusField::Birth
                                             ? FocusField::Survival
                                             : FocusField::Birth);
                }
                if (keysym == XK_Escape) {
                    return set_focus(FocusField::Inactive);
                }
                if ((text_length > 0) && (text[0] >= '0') && (text[0] <= '8')) {
                    std::string *input = focused_input();
                    return (input != nullptr) && append_unique_digit(*input, text[0]);
                }
                return false;
            }

            switch (keysym) {
                case XK_Left:
                case XK_a:
                case XK_h:
                    return pan_viewport(-kViewportPanStep, 0);
                case XK_Right:
                case XK_d:
                case XK_l:
                    return pan_viewport(kViewportPanStep, 0);
                case XK_Up:
                case XK_w:
                case XK_k:
                    return pan_viewport(0, -kViewportPanStep);
                case XK_Down:
                case XK_s:
                case XK_j:
                    return pan_viewport(0, kViewportPanStep);
                case XK_Home:
                    return set_viewport(0, 0);
                default:
                    return false;
            }
        }

        [[nodiscard]] bool handle_button_press(const XButtonEvent &event) {
            switch (event.button) {
                case Button1: {
                    const bool focus_cleared =
                            (_focused_field != FocusField::Inactive) &&
                            !_panel_rect.contains(event.x, event.y) &&
                            set_focus(FocusField::Inactive);
                    if (_panel_rect.contains(event.x, event.y)) {
                        if (_birth_field_rect.contains(event.x, event.y)) {
                            return set_focus(FocusField::Birth);
                        }
                        if (_survival_field_rect.contains(event.x, event.y)) {
                            return set_focus(FocusField::Survival);
                        }
                        if (_apply_button_rect.contains(event.x, event.y)) {
                            (void) set_focus(FocusField::Inactive);
                            return queue_apply_from_inputs();
                        }
                        if (_reset_button_rect.contains(event.x, event.y)) {
                            (void) set_focus(FocusField::Inactive);
                            return reset_to_default_rules();
                        }
                        return set_focus(FocusField::Inactive);
                    }
                    if (_horizontal_thumb_rect.contains(event.x, event.y)) {
                        _drag_mode = DragMode::HorizontalScrollbar;
                        _drag_pointer_offset = event.x - _horizontal_thumb_rect.x;
                        return focus_cleared;
                    }
                    if (_vertical_thumb_rect.contains(event.x, event.y)) {
                        _drag_mode = DragMode::VerticalScrollbar;
                        _drag_pointer_offset = event.y - _vertical_thumb_rect.y;
                        return focus_cleared;
                    }
                    if (_horizontal_scrollbar_rect.contains(event.x, event.y)) {
                        return page_viewport(event.x < _horizontal_thumb_rect.x ? -1 : 1, 0) || focus_cleared;
                    }
                    if (_vertical_scrollbar_rect.contains(event.x, event.y)) {
                        return page_viewport(0, event.y < _vertical_thumb_rect.y ? -1 : 1) || focus_cleared;
                    }
                    return focus_cleared;
                }
                case Button2:
                    if (_content_rect.contains(event.x, event.y)) {
                        _drag_mode = DragMode::MiddlePan;
                        _drag_anchor_x = event.x;
                        _drag_anchor_y = event.y;
                        _drag_viewport_origin_x = _viewport_x;
                        _drag_viewport_origin_y = _viewport_y;
                    }
                    return false;
                case Button4:
                case Button5:
                case 6:
                case 7:
                    return false;
                default:
                    return false;
            }
        }

        [[nodiscard]] bool handle_button_release(const XButtonEvent &event) {
            if ((event.button == Button1) &&
                ((_drag_mode == DragMode::HorizontalScrollbar) || (_drag_mode == DragMode::VerticalScrollbar))) {
                _drag_mode = DragMode::Idle;
            }
            if ((event.button == Button2) && (_drag_mode == DragMode::MiddlePan)) {
                _drag_mode = DragMode::Idle;
            }
            return false;
        }

        [[nodiscard]] bool handle_motion_notify(const XMotionEvent &event) {
            switch (_drag_mode) {
                case DragMode::HorizontalScrollbar: {
                    const int thumb_offset = event.x - _drag_pointer_offset - _horizontal_scrollbar_rect.x;
                    return set_viewport(
                            compute_viewport_offset_from_thumb(
                                    thumb_offset,
                                    max_viewport_x(),
                                    _horizontal_scrollbar_rect.width,
                                    _horizontal_thumb_rect.width),
                            _viewport_y);
                }
                case DragMode::VerticalScrollbar: {
                    const int thumb_offset = event.y - _drag_pointer_offset - _vertical_scrollbar_rect.y;
                    return set_viewport(
                            _viewport_x,
                            compute_viewport_offset_from_thumb(
                                    thumb_offset,
                                    max_viewport_y(),
                                    _vertical_scrollbar_rect.height,
                                    _vertical_thumb_rect.height));
                }
                case DragMode::MiddlePan:
                    return set_viewport(_drag_viewport_origin_x - (event.x - _drag_anchor_x),
                                        _drag_viewport_origin_y - (event.y - _drag_anchor_y));
                case DragMode::Idle:
                default:
                    return false;
            }
        }

        Display *_display = nullptr;
        Window _window = 0;
        GC _gc = nullptr;
        Atom _wm_delete = 0;
        XImage *_image = nullptr;
        XShmSegmentInfo _shm_info{};
        PixelPalette _palette{};
        UiPalette _ui_palette{};
        Rect _content_rect{};
        Rect _panel_rect{};
        Rect _horizontal_scrollbar_rect{};
        Rect _horizontal_thumb_rect{};
        Rect _vertical_scrollbar_rect{};
        Rect _vertical_thumb_rect{};
        Rect _corner_rect{};
        Rect _birth_field_rect{};
        Rect _survival_field_rect{};
        Rect _apply_button_rect{};
        Rect _reset_button_rect{};
        int _pitch_bytes = 0;
        int _image_width = 0;
        int _image_height = 0;
        int _window_width = 0;
        int _window_height = 0;
        int _viewport_x = 0;
        int _viewport_y = 0;
        int _drag_pointer_offset = 0;
        int _drag_anchor_x = 0;
        int _drag_anchor_y = 0;
        int _drag_viewport_origin_x = 0;
        int _drag_viewport_origin_y = 0;
        DragMode _drag_mode = DragMode::Idle;
        FocusField _focused_field = FocusField::Inactive;
        std::string _birth_input = RuleSet::conway().birth_digits();
        std::string _survival_input = RuleSet::conway().survive_digits();
        std::string _current_rule_label = RuleSet::conway().format();
        RuleSet _pending_rules = RuleSet::conway();
        bool _has_pending_rule_change = false;
        bool _show_horizontal_scrollbar = false;
        bool _show_vertical_scrollbar = false;
        bool _use_shm = false;
        bool _stream_stores = false;
    };

    [[nodiscard]] HeadlessFrameBuffer create_headless_frame_buffer(int width, int height) {
        HeadlessFrameBuffer buffer;
        buffer.pitch_bytes = width * static_cast<int>(sizeof(std::uint32_t));
        buffer.bytes = allocate_aligned_bytes(
                static_cast<std::size_t>(buffer.pitch_bytes) * static_cast<std::size_t>(height),
                64);
        buffer.palette = {0U, 0x00FFFFFFU};
        buffer.stream_stores =
                buffer.bytes &&
                (((static_cast<std::uintptr_t>(buffer.pitch_bytes) |
                   reinterpret_cast<std::uintptr_t>(buffer.bytes.get())) & 31U) == 0U);
        return buffer;
    }
}

class LifeBoard::Impl {
public:
    Impl(std::shared_ptr<CellSet> board,
         int threads,
         int width,
         int height,
         Backend backend,
         LifeBoard::RuleSet rules)
            : _board(std::move(board), threads, width, height, backend, rules) {
    }

    Impl(CellBuffer cells,
         int threads,
         int width,
         int height,
         Backend backend,
         LifeBoard::RuleSet rules)
            : _board(std::move(cells), threads, width, height, backend, rules) {
    }

    [[nodiscard]] FrameView iterate(const RenderTarget *render_target) {
        if (_return_current_without_step) {
            _return_current_without_step = false;
            if (render_target != nullptr) {
                _board.render_current(*render_target);
            }
        } else if (render_target != nullptr) {
            _board.step_and_render(*render_target);
        } else {
            _board.step();
        }

        _board.materialize(_frame_cache);
        return {_frame_cache.empty() ? nullptr : &_frame_cache,
                _board.width(),
                _board.height(),
                _board.width() + 2,
                _board.width() + 3};
    }

    void advance(const RenderTarget *render_target) {
        if (render_target != nullptr) {
            _board.step_and_render(*render_target);
        } else {
            _board.step();
        }
        _return_current_without_step = true;
    }

    [[nodiscard]] FrameView render(const RenderTarget &render_target) {
        _board.render_current(render_target);
        _board.materialize(_frame_cache);
        _return_current_without_step = true;
        return {_frame_cache.empty() ? nullptr : &_frame_cache,
                _board.width(),
                _board.height(),
                _board.width() + 2,
                _board.width() + 3};
    }

    [[nodiscard]] CellSet snapshot() const {
        return _board.snapshot();
    }

    [[nodiscard]] bool alive(Cell cell) const {
        return _board.alive(cell);
    }

    void set_rules(LifeBoard::RuleSet rules) {
        _board.set_rules(rules);
    }

    [[nodiscard]] LifeBoard::RuleSet rules() const {
        return _board.rules();
    }

    [[nodiscard]] Backend backend() const {
        return _board.backend();
    }

    [[nodiscard]] int width() const {
        return _board.width();
    }

    [[nodiscard]] int height() const {
        return _board.height();
    }

private:
    SimulationBoard _board;
    CellBuffer _frame_cache;
    bool _return_current_without_step = true;
};

LifeBoard::LifeBoard(std::shared_ptr<CellSet> board,
                     int threads,
                     int width,
                     int height,
                     Backend backend,
                     LifeBoard::RuleSet rules)
        : _impl(std::make_unique<Impl>(std::move(board), threads, width, height, backend, rules)) {
}

LifeBoard::LifeBoard(CellBuffer cells,
                     int threads,
                     int width,
                     int height,
                     Backend backend,
                     LifeBoard::RuleSet rules)
        : _impl(std::make_unique<Impl>(std::move(cells), threads, width, height, backend, rules)) {
}

LifeBoard::~LifeBoard() = default;

LifeBoard::FrameView LifeBoard::iterate() {
    return _impl->iterate(nullptr);
}

LifeBoard::FrameView LifeBoard::iterate(const RenderTarget &render_target) {
    return _impl->iterate(&render_target);
}

void LifeBoard::advance() {
    _impl->advance(nullptr);
}

void LifeBoard::advance(const RenderTarget &render_target) {
    _impl->advance(&render_target);
}

LifeBoard::FrameView LifeBoard::render(const RenderTarget &render_target) {
    return _impl->render(render_target);
}

LifeBoard::CellSet LifeBoard::snapshot() const {
    return _impl->snapshot();
}

bool LifeBoard::alive(Cell cell) const {
    return _impl->alive(cell);
}

void LifeBoard::set_rules(LifeBoard::RuleSet rules) {
    _impl->set_rules(rules);
}

LifeBoard::RuleSet LifeBoard::rules() const {
    return _impl->rules();
}

LifeBoard::Backend LifeBoard::backend() const {
    return _impl->backend();
}

const char *LifeBoard::backend_name() const {
    return backend_name_for_enum(_impl->backend());
}

int LifeBoard::width() const {
    return _impl->width();
}

int LifeBoard::height() const {
    return _impl->height();
}

#ifndef CLIFE_NO_MAIN
int main(int argc, char **args) {
    (void) argc;
    (void) args;

    const RuntimeOptions options = load_runtime_options();
    const bool avx2_available = detect_avx2();
    const bool use_render_avx2 =
            options.render_backend == RenderBackend::Scalar ? false :
            options.render_backend == RenderBackend::Avx2 ? avx2_available :
            avx2_available;
    const int thread_count = select_thread_count(options.threads, options.width, options.height);
    const Backend runtime_backend = Backend::BitPacked;

    SimulationBoard board(make_seed_cells(options.width, options.height, options.density, options.seed),
                          thread_count,
                          options.width,
                          options.height,
                          runtime_backend);

    HeadlessFrameBuffer headless_buffer;
    X11FrameBuffer x11_buffer;
    std::uint8_t *surface_bytes = nullptr;
    int pitch_bytes = 0;
    PixelPalette palette{};
    bool stream_stores = false;
    PixelLookupTable pixel_lut;
    RenderTarget render_target{};
    const bool needs_render_target =
            (options.benchmark_frames == 0) ||
            (options.benchmark_mode != BenchMode::Update);

    if (needs_render_target) {
        if (options.benchmark_frames > 0) {
            headless_buffer = create_headless_frame_buffer(options.width, options.height);
            if (!headless_buffer.bytes) {
                std::cerr << "Failed to allocate benchmark frame buffer" << std::endl;
                return EXIT_FAILURE;
            }
            surface_bytes = headless_buffer.bytes.get();
            pitch_bytes = headless_buffer.pitch_bytes;
            palette = headless_buffer.palette;
            stream_stores = headless_buffer.stream_stores;
        } else {
            if (!x11_buffer.create(options.width, options.height)) {
                return EXIT_FAILURE;
            }
            surface_bytes = x11_buffer.pixels();
            pitch_bytes = x11_buffer.pitch_bytes();
            palette = x11_buffer.palette();
            stream_stores = x11_buffer.stream_stores();
        }

        pixel_lut = make_pixel_lookup_table(palette.dead, palette.alive);
        render_target = {
                surface_bytes,
                pitch_bytes,
                palette.dead,
                palette.alive,
                pixel_lut.entries.data(),
                stream_stores,
                use_render_avx2,
        };
    }

    if (options.benchmark_frames > 0) {
        auto run_benchmark_iteration = [&](int &simulated_steps, int &rendered_frames) {
            switch (options.benchmark_mode) {
                case BenchMode::Update:
                    board.step();
                    ++simulated_steps;
                    break;

                case BenchMode::Render:
                    board.render_current(render_target);
                    ++rendered_frames;
                    break;

                case BenchMode::Combined:
                default:
                    board.step_and_render(render_target);
                    ++simulated_steps;
                    ++rendered_frames;
                    break;
            }
        };

        int warmup_steps = 0;
        int warmup_renders = 0;
        for (int frame = 0; frame < options.benchmark_warmup; ++frame) {
            run_benchmark_iteration(warmup_steps, warmup_renders);
        }

        int simulated_steps = 0;
        int rendered_frames = 0;
        const auto benchmark_start = std::chrono::steady_clock::now();

        if (options.benchmark_min_seconds > 0.0) {
            int measured_iterations = 0;
            while ((measured_iterations < options.benchmark_frames) ||
                   (std::chrono::duration_cast<std::chrono::duration<double>>(
                            std::chrono::steady_clock::now() - benchmark_start).count() <
                    options.benchmark_min_seconds)) {
                run_benchmark_iteration(simulated_steps, rendered_frames);
                ++measured_iterations;
            }
        } else {
            for (int frame = 0; frame < options.benchmark_frames; ++frame) {
                run_benchmark_iteration(simulated_steps, rendered_frames);
            }
        }

        const auto benchmark_end = std::chrono::steady_clock::now();
        const double seconds =
                std::chrono::duration_cast<std::chrono::duration<double>>(benchmark_end - benchmark_start).count();
        const double updates_per_second = seconds > 0.0 ? static_cast<double>(simulated_steps) / seconds : 0.0;
        const double renders_per_second = seconds > 0.0 ? static_cast<double>(rendered_frames) / seconds : 0.0;
        const double cell_updates_per_second =
                seconds > 0.0
                        ? (static_cast<double>(simulated_steps) *
                           static_cast<double>(options.width) *
                           static_cast<double>(options.height)) / seconds
                        : 0.0;

        std::cout << "mode=" << bench_mode_name(options.benchmark_mode)
                  << " backend=" << backend_name_for_enum(board.backend())
                  << " render_backend=" << (needs_render_target
                          ? (use_render_avx2 ? "avx2" : "scalar")
                          : "none")
                  << " width=" << options.width
                  << " height=" << options.height
                  << " density=" << options.density
                  << " seed=" << options.seed
                  << " threads=" << thread_count
                  << " benchmark_warmup=" << options.benchmark_warmup
                  << " benchmark_frames=" << options.benchmark_frames
                  << " benchmark_min_seconds=" << options.benchmark_min_seconds
                  << " simulated_steps=" << simulated_steps
                  << " rendered_frames=" << rendered_frames
                  << " elapsed_seconds=" << seconds
                  << " updates_per_second=" << updates_per_second
                  << " renders_per_second=" << renders_per_second
                  << " cell_updates_per_second=" << cell_updates_per_second
                  << std::endl;
        return EXIT_SUCCESS;
    }

    constexpr auto kRepaintInterval = std::chrono::microseconds(1'000'000 / 30);
    constexpr auto kIdleSleep = std::chrono::milliseconds(1);

    std::mutex render_mutex;
    std::atomic<bool> repaint_requested{true};
    std::atomic<std::uint64_t> painted_generation{0};

    std::jthread simulation_thread([&](std::stop_token stop_token) {
        bool first_frame = true;
        while (!stop_token.stop_requested()) {
            if (repaint_requested.exchange(false, std::memory_order_acq_rel)) {
                {
                    std::scoped_lock lock(render_mutex);
                    if (first_frame) {
                        board.render_current(render_target);
                        first_frame = false;
                    } else {
                        board.step_and_render(render_target);
                    }
                }
                painted_generation.fetch_add(1, std::memory_order_release);
            } else if (first_frame) {
                first_frame = false;
            } else {
                board.step();
            }
        }
    });

    bool running = true;
    std::uint64_t presented_generation = 0;
    auto next_repaint_request = std::chrono::steady_clock::now() + kRepaintInterval;

    while (running) {
        bool redraw_requested = x11_buffer.poll_events(running);
        if (!running) {
            break;
        }

        RuleSet requested_rules;
        if (x11_buffer.consume_rule_change(requested_rules)) {
            board.set_rules(requested_rules);
            redraw_requested = true;
        }
        if (x11_buffer.set_current_rules(board.rules())) {
            redraw_requested = true;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now >= next_repaint_request) {
            repaint_requested.store(true, std::memory_order_release);
            do {
                next_repaint_request += kRepaintInterval;
            } while (now >= next_repaint_request);
        }

        const std::uint64_t available_generation = painted_generation.load(std::memory_order_acquire);
        if ((available_generation != presented_generation) || redraw_requested) {
            std::scoped_lock lock(render_mutex);
            const std::uint64_t confirmed_generation = painted_generation.load(std::memory_order_acquire);
            if ((confirmed_generation != presented_generation) || redraw_requested) {
                x11_buffer.present();
                presented_generation = confirmed_generation;
            }
        } else {
            std::this_thread::sleep_for(kIdleSleep);
        }
    }

    return EXIT_SUCCESS;
}
#endif

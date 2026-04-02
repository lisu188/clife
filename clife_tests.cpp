/*
 * MIT License
 *
 * Copyright (c) 2019 Andrzej Lis
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "clife.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>


namespace {
    [[nodiscard]] const char *backend_name(LifeBoard::Backend backend) {
        switch (backend) {
            case LifeBoard::Backend::Byte:
                return "byte";
            case LifeBoard::Backend::BitPacked:
                return "bitpacked";
            case LifeBoard::Backend::Reference:
                return "reference";
            case LifeBoard::Backend::Auto:
            default:
                return "auto";
        }
    }

    [[nodiscard]] int wrap_coordinate(int value, int extent) {
        const int wrapped = value % extent;
        return wrapped >= 0 ? wrapped : wrapped + extent;
    }

    [[nodiscard]] LifeBoard::CellBuffer step_reference(const LifeBoard::CellBuffer &cells,
                                                       int width,
                                                       int height,
                                                       LifeBoard::RuleSet rules) {
        const auto rule_lut = rules.normalized().scalar_lookup();
        LifeBoard::CellBuffer next(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int neighbors = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if ((dx == 0) && (dy == 0)) {
                            continue;
                        }
                        const int nx = wrap_coordinate(x + dx, width);
                        const int ny = wrap_coordinate(y + dy, height);
                        neighbors += cells[static_cast<std::size_t>(ny) * static_cast<std::size_t>(width) + static_cast<std::size_t>(nx)] != 0 ? 1 : 0;
                    }
                }
                const std::uint8_t alive = cells[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)];
                next[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] =
                        rule_lut[static_cast<std::size_t>((alive != 0 ? 9 : 0) + neighbors)];
            }
        }
        return next;
    }

    [[nodiscard]] LifeBoard::CellBuffer normalize_view(const LifeBoard::FrameView &view) {
        LifeBoard::CellBuffer cells(static_cast<std::size_t>(view.view_width) * static_cast<std::size_t>(view.view_height), 0);
        const auto &source = *view.cells;
        for (int y = 0; y < view.view_height; ++y) {
            const int row_index = view.top_left_index + y * view.stride;
            std::copy_n(source.data() + row_index,
                        view.view_width,
                        cells.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(view.view_width));
        }
        return cells;
    }

    void require(bool condition, std::string_view message) {
        if (!condition) {
            throw std::runtime_error(std::string(message));
        }
    }

    void require_equal(const LifeBoard::CellBuffer &lhs, const LifeBoard::CellBuffer &rhs, std::string_view message) {
        if (lhs != rhs) {
            throw std::runtime_error(std::string(message));
        }
    }

    [[nodiscard]] std::string cells_to_string(const LifeBoard::CellBuffer &cells, int width, int height) {
        std::string out;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                out.push_back(cells[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                    static_cast<std::size_t>(x)] != 0 ? '1' : '0');
            }
            if (y + 1 != height) {
                out.push_back('/');
            }
        }
        return out;
    }

    [[nodiscard]] std::uint32_t pixel_for_alive(bool alive) {
        return alive ? 0x00FFFFFFU : 0U;
    }

    [[nodiscard]] LifeBoard::CellSet cells_to_set(const LifeBoard::CellBuffer &cells, int width, int height) {
        LifeBoard::CellSet board;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (cells[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                          static_cast<std::size_t>(x)] != 0) {
                    board.emplace(x, y);
                }
            }
        }
        return board;
    }

    void verify_render_buffer(
            const std::vector<std::uint32_t> &pixels,
            const LifeBoard::CellBuffer &cells,
            int width,
            int height,
            int pitch_pixels,
            std::string_view message) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::uint32_t expected = pixel_for_alive(cells[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] != 0);
                const std::uint32_t actual =
                        pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch_pixels) +
                               static_cast<std::size_t>(x)];
                if (actual != expected) {
                    throw std::runtime_error(std::string(message));
                }
            }
        }
    }

    [[nodiscard]] LifeBoard::CellBuffer make_case_cells(int width, int height, std::uint64_t seed, float density) {
        LifeBoard::CellBuffer cells(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
        std::uint64_t state = seed == 0 ? 0xA5A5A5A5A5A5A5A5ULL : seed;
        const std::uint32_t threshold = static_cast<std::uint32_t>(
                std::clamp(static_cast<double>(density), 0.0, 1.0) * static_cast<double>(std::numeric_limits<std::uint32_t>::max()));
        for (std::uint8_t &cell: cells) {
            state ^= state >> 12U;
            state ^= state << 25U;
            state ^= state >> 27U;
            const std::uint32_t sample = static_cast<std::uint32_t>((state * 2685821657736338717ULL) >> 32U);
            cell = static_cast<std::uint8_t>(sample <= threshold);
        }
        return cells;
    }

    [[nodiscard]] std::array<std::uint32_t, 256 * 8> make_pixel_lut() {
        std::array<std::uint32_t, 256 * 8> lut{};
        for (std::size_t pattern = 0; pattern < 256; ++pattern) {
            for (int bit = 0; bit < 8; ++bit) {
                lut[pattern * 8 + static_cast<std::size_t>(bit)] = pixel_for_alive(((pattern >> bit) & 1U) != 0U);
            }
        }
        return lut;
    }

    void run_backend_case(const LifeBoard::CellBuffer &initial,
                          int width,
                          int height,
                          LifeBoard::Backend requested_backend,
                          LifeBoard::RuleSet rules) {
        const std::string case_prefix = std::string(backend_name(requested_backend)) +
                                        " " + rules.format() +
                                        " " + std::to_string(width) + "x" + std::to_string(height) + ": ";
        LifeBoard board(initial, 4, width, height, requested_backend, rules);
        require(board.backend() == LifeBoard::Backend::BitPacked,
                case_prefix + "all runtime backends must resolve to bitpacked");
        require(board.rules() == rules.normalized(), case_prefix + "initial rules mismatch");
        const LifeBoard::FrameView initial_view = board.iterate();
        require_equal(normalize_view(initial_view), initial, case_prefix + "first iterate() must expose the initial state");

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                require(board.alive({x, y}) == (initial[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] != 0),
                        case_prefix + "alive() mismatch on initial state");
            }
        }

        const auto expected_after_one = step_reference(initial, width, height, rules);
        const LifeBoard::FrameView stepped_view = board.iterate();
        const auto stepped_cells = normalize_view(stepped_view);
        if (stepped_cells != expected_after_one) {
            throw std::runtime_error(case_prefix + "second iterate() must advance once"
                                     + " expected=" + cells_to_string(expected_after_one, width, height)
                                     + " actual=" + cells_to_string(stepped_cells, width, height));
        }
        require(board.snapshot() == cells_to_set(expected_after_one, width, height),
                case_prefix + "snapshot() mismatch");

        const auto expected_after_two = step_reference(expected_after_one, width, height, rules);
        board.advance();
        require_equal(normalize_view(board.iterate()), expected_after_two, case_prefix + "advance() must step even before iterate()");

        const auto pixel_lut = make_pixel_lut();
        const int pitch_pixels = width + 3;
        std::vector<std::uint32_t> pixels(
                static_cast<std::size_t>(pitch_pixels) * static_cast<std::size_t>(height),
                0xDEADBEEFU);
        const LifeBoard::RenderTarget render_target = {
                reinterpret_cast<std::uint8_t *>(pixels.data()),
                pitch_pixels * static_cast<int>(sizeof(std::uint32_t)),
                0U,
                0x00FFFFFFU,
                pixel_lut.data(),
                false,
                false,
        };

        LifeBoard render_board(initial, 4, width, height, requested_backend, rules);
        const LifeBoard::FrameView rendered_initial = render_board.iterate(render_target);
        require_equal(normalize_view(rendered_initial), initial, case_prefix + "first iterate(render) must expose the initial state");
        verify_render_buffer(pixels, initial, width, height, pitch_pixels, case_prefix + "first iterate(render) buffer mismatch");

        const auto expected_render_step = step_reference(initial, width, height, rules);
        const LifeBoard::FrameView rendered_next = render_board.iterate(render_target);
        require_equal(normalize_view(rendered_next), expected_render_step, case_prefix + "second iterate(render) must advance once");
        verify_render_buffer(pixels, expected_render_step, width, height, pitch_pixels, case_prefix + "second iterate(render) buffer mismatch");

        LifeBoard direct_render_board(initial, 4, width, height, requested_backend, rules);
        std::fill(pixels.begin(), pixels.end(), 0U);
        direct_render_board.render(render_target);
        verify_render_buffer(pixels, initial, width, height, pitch_pixels, case_prefix + "render() buffer mismatch");

        LifeBoard fast_advance_board(initial, 4, width, height, requested_backend, rules);
        fast_advance_board.advance(render_target);
        verify_render_buffer(pixels, expected_after_one, width, height, pitch_pixels, case_prefix + "advance(render) buffer mismatch");
        require_equal(normalize_view(fast_advance_board.iterate()), expected_after_one, case_prefix + "iterate() after advance(render) must show the current state");
    }

    void run_rule_change_case(const LifeBoard::CellBuffer &initial, int width, int height, LifeBoard::Backend requested_backend) {
        const LifeBoard::RuleSet conway = LifeBoard::RuleSet::conway();
        const LifeBoard::RuleSet highlife = LifeBoard::RuleSet::from_digit_strings("36", "23");
        const LifeBoard::RuleSet day_night = LifeBoard::RuleSet::from_digit_strings("3678", "34678");
        LifeBoard board(initial, 4, width, height, requested_backend, conway);
        require(board.backend() == LifeBoard::Backend::BitPacked,
                std::string(backend_name(requested_backend)) + " runtime rule-change backend mismatch");
        require_equal(normalize_view(board.iterate()), initial, "runtime rule-change initial iterate mismatch");

        const auto after_conway = step_reference(initial, width, height, conway);
        board.advance();
        require_equal(normalize_view(board.iterate()), after_conway, "runtime rule-change first step mismatch");

        board.set_rules(highlife);
        const auto after_highlife = step_reference(after_conway, width, height, highlife);
        board.advance();
        require(board.rules() == highlife.normalized(), "runtime rule-change must publish new active rules after stepping");
        require_equal(normalize_view(board.iterate()), after_highlife, "runtime rule-change second step mismatch");

        const LifeBoard::RuleSet seeds = LifeBoard::RuleSet::from_digit_strings("2", "");
        board.set_rules(seeds);
        const auto after_seeds = step_reference(after_highlife, width, height, seeds);
        board.advance();
        require(board.rules() == seeds.normalized(), "runtime rule-change seeds rules mismatch");
        require_equal(normalize_view(board.iterate()), after_seeds, "runtime rule-change seeds step mismatch");

        board.set_rules(day_night);
        const auto after_day_night = step_reference(after_seeds, width, height, day_night);
        board.advance();
        require(board.rules() == day_night.normalized(), "runtime rule-change day-and-night rules mismatch");
        require_equal(normalize_view(board.iterate()), after_day_night, "runtime rule-change day-and-night step mismatch");
    }
}

int main() {
    try {
        const LifeBoard::RuleSet conway = LifeBoard::RuleSet::conway();
        const LifeBoard::RuleSet highlife = LifeBoard::RuleSet::from_digit_strings("36", "23");
        const LifeBoard::RuleSet twox2 = LifeBoard::RuleSet::from_digit_strings("36", "125");
        const LifeBoard::RuleSet day_night = LifeBoard::RuleSet::from_digit_strings("3678", "34678");
        const LifeBoard::RuleSet morley = LifeBoard::RuleSet::from_digit_strings("368", "245");
        const LifeBoard::RuleSet seeds = LifeBoard::RuleSet::from_digit_strings("2", "");
        const std::vector<std::pair<int, int>> dimensions = {
                {1, 1},
                {1, 5},
                {2, 2},
                {3, 3},
                {5, 7},
                {17, 19},
                {64, 64},
                {65, 17},
                {130, 33},
        };
        const std::array<float, 5> densities = {0.0F, 0.05F, 0.3F, 0.8F, 1.0F};
        const std::array<LifeBoard::Backend, 4> backend_requests = {
                LifeBoard::Backend::Reference,
                LifeBoard::Backend::Byte,
                LifeBoard::Backend::BitPacked,
                LifeBoard::Backend::Auto,
        };

        for (const auto &[width, height]: dimensions) {
            for (float density: densities) {
                for (std::uint64_t seed = 1; seed <= 3; ++seed) {
                    const auto initial = make_case_cells(width, height, seed, density);
                    for (LifeBoard::Backend requested_backend: backend_requests) {
                        run_backend_case(initial, width, height, requested_backend, conway);
                    }
                }
            }
        }

        LifeBoard::CellBuffer blinker = {
                0, 0, 0, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 0, 0, 0,
        };
        for (LifeBoard::Backend requested_backend: backend_requests) {
            run_backend_case(blinker, 5, 5, requested_backend, conway);
        }

        const auto highlife_case = make_case_cells(17, 19, 11, 0.32F);
        const auto twox2_case = make_case_cells(17, 19, 13, 0.28F);
        const auto day_night_case = make_case_cells(17, 19, 19, 0.44F);
        const auto morley_case = make_case_cells(17, 19, 23, 0.26F);
        const auto seeds_case = make_case_cells(17, 19, 17, 0.18F);
        for (LifeBoard::Backend requested_backend: backend_requests) {
            run_backend_case(highlife_case, 17, 19, requested_backend, highlife);
            run_backend_case(twox2_case, 17, 19, requested_backend, twox2);
            run_backend_case(day_night_case, 17, 19, requested_backend, day_night);
            run_backend_case(morley_case, 17, 19, requested_backend, morley);
            run_backend_case(seeds_case, 17, 19, requested_backend, seeds);
        }

        LifeBoard::CellBuffer runtime_case = {
                0, 1, 0, 0, 1,
                1, 0, 1, 0, 0,
                0, 1, 1, 1, 0,
                0, 0, 0, 1, 0,
                1, 0, 0, 0, 1,
        };
        for (LifeBoard::Backend requested_backend: backend_requests) {
            run_rule_change_case(runtime_case, 5, 5, requested_backend);
        }
    } catch (const std::exception &ex) {
        std::cerr << "clife_tests: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "clife_tests: ok" << std::endl;
    return EXIT_SUCCESS;
}

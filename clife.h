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
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>


class LifeBoard {
public:
    using Cell = std::pair<int, int>;
    struct CellHash {
        std::size_t operator()(const Cell &cell) const noexcept {
            return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(cell.first)) << 32) ^
                   static_cast<std::uint32_t>(cell.second);
        }
    };
    using CellSet = std::unordered_set<Cell, CellHash>;
    using IndexList = std::vector<int>;
    using CellBuffer = std::vector<std::uint8_t>;

    enum class Backend {
        Auto,
        Byte,
        BitPacked,
        Reference,
    };

    struct RuleSet {
        static constexpr std::uint16_t kMaskBits = (1U << 9U) - 1U;

        std::uint16_t birth_mask = static_cast<std::uint16_t>(1U << 3U);
        std::uint16_t survive_mask = static_cast<std::uint16_t>((1U << 2U) | (1U << 3U));

        [[nodiscard]] static constexpr RuleSet conway() noexcept {
            return {};
        }

        [[nodiscard]] static constexpr std::uint16_t normalize_mask(std::uint16_t mask) noexcept {
            return static_cast<std::uint16_t>(mask & kMaskBits);
        }

        [[nodiscard]] constexpr RuleSet normalized() const noexcept {
            return {normalize_mask(birth_mask), normalize_mask(survive_mask)};
        }

        [[nodiscard]] constexpr bool is_conway() const noexcept {
            const RuleSet rule = normalized();
            return (rule.birth_mask == conway().birth_mask) &&
                   (rule.survive_mask == conway().survive_mask);
        }

        [[nodiscard]] static constexpr std::uint16_t parse_digit_mask(std::string_view digits) noexcept {
            std::uint16_t mask = 0;
            for (char ch: digits) {
                if ((ch >= '0') && (ch <= '8')) {
                    mask = static_cast<std::uint16_t>(mask | (1U << static_cast<unsigned>(ch - '0')));
                }
            }
            return mask;
        }

        [[nodiscard]] static constexpr RuleSet from_digit_strings(std::string_view birth_digits,
                                                                  std::string_view survive_digits) noexcept {
            return {parse_digit_mask(birth_digits), parse_digit_mask(survive_digits)};
        }

        [[nodiscard]] static std::string digits_for_mask(std::uint16_t mask) {
            std::string digits;
            mask = normalize_mask(mask);
            for (int neighbor_count = 0; neighbor_count <= 8; ++neighbor_count) {
                if (((mask >> neighbor_count) & 1U) != 0U) {
                    digits.push_back(static_cast<char>('0' + neighbor_count));
                }
            }
            return digits;
        }

        [[nodiscard]] static std::string normalize_digits(std::string_view digits) {
            return digits_for_mask(parse_digit_mask(digits));
        }

        [[nodiscard]] std::string birth_digits() const {
            return digits_for_mask(normalized().birth_mask);
        }

        [[nodiscard]] std::string survive_digits() const {
            return digits_for_mask(normalized().survive_mask);
        }

        [[nodiscard]] std::string format() const {
            const RuleSet rule = normalized();
            return std::string("B") + digits_for_mask(rule.birth_mask) +
                   "/S" + digits_for_mask(rule.survive_mask);
        }

        [[nodiscard]] std::array<std::uint8_t, 18> scalar_lookup() const noexcept {
            const RuleSet rule = normalized();
            std::array<std::uint8_t, 18> lut{};
            for (int alive = 0; alive <= 1; ++alive) {
                for (int neighbors = 0; neighbors <= 8; ++neighbors) {
                    const bool born =
                            (alive == 0) && (((rule.birth_mask >> neighbors) & 1U) != 0U);
                    const bool survives =
                            (alive != 0) && (((rule.survive_mask >> neighbors) & 1U) != 0U);
                    lut[static_cast<std::size_t>(alive * 9 + neighbors)] =
                            static_cast<std::uint8_t>(born || survives);
                }
            }
            return lut;
        }

        [[nodiscard]] constexpr bool operator==(const RuleSet &) const noexcept = default;
    };

    struct FrameView {
        const std::vector<std::uint8_t> *cells = nullptr;
        int view_width = 0;
        int view_height = 0;
        int stride = 0;
        int top_left_index = 0;
    };

    struct RenderTarget {
        std::uint8_t *surface_bytes = nullptr;
        int pitch_bytes = 0;
        std::uint32_t dead_pixel = 0;
        std::uint32_t alive_pixel = 0x00FFFFFFU;
        const std::uint32_t *pixel_lut = nullptr;
        bool stream_stores = false;
        bool use_avx2 = false;
    };

    LifeBoard(std::shared_ptr<CellSet> board,
              int threads,
              int width = 0,
              int height = 0,
              Backend backend = Backend::Auto,
              RuleSet rules = RuleSet::conway());
    LifeBoard(CellBuffer cells,
              int threads,
              int width,
              int height,
              Backend backend = Backend::Auto,
              RuleSet rules = RuleSet::conway());
    ~LifeBoard();

    LifeBoard(const LifeBoard &) = delete;
    LifeBoard &operator=(const LifeBoard &) = delete;

    void advance();
    void advance(const RenderTarget &render_target);
    FrameView render(const RenderTarget &render_target);

    FrameView iterate();
    FrameView iterate(const RenderTarget &render_target);

    CellSet snapshot() const;

    bool alive(Cell cell) const;

    void set_rules(RuleSet rules);

    [[nodiscard]] RuleSet rules() const;

    int width() const;

    int height() const;

    Backend backend() const;

    const char *backend_name() const;

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};

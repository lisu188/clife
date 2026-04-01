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

#include <SDL.h>
#include <cstdint>
#include <memory>
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

    struct DirtySpan {
        int row = 0;
        int min_x = 0;
        int max_x = -1;
    };

    struct FrameView {
        const std::vector<std::uint8_t> *cells = nullptr;
        const std::vector<DirtySpan> *dirty_spans = nullptr;
        int view_width = 0;
        int view_height = 0;
        int stride = 0;
        int top_left_index = 0;
        bool full_refresh = false;
    };

    LifeBoard(std::shared_ptr<CellSet> board, int threads, int width = 0, int height = 0);
    LifeBoard(CellBuffer cells, int threads, int width, int height);
    ~LifeBoard();

    LifeBoard(const LifeBoard &) = delete;
    LifeBoard &operator=(const LifeBoard &) = delete;

    FrameView iterate();

    CellSet snapshot() const;

    bool alive(Cell cell) const;

    int width() const;

    int height() const;

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};

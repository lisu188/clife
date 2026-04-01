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
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>
#include <vstd.h>


class LifeBoard {
public:
    using Cell = std::pair<int, int>;
    using CellSet = std::unordered_set<Cell>;
    using CellList = std::vector<Cell>;

    LifeBoard(std::shared_ptr<CellSet> board, int threads);

    ~LifeBoard();

    std::shared_ptr<const CellSet> iterate();

    std::shared_ptr<CellSet> _prev_board = std::make_shared<CellSet>();
    std::shared_ptr<CellSet> _next_board = std::make_shared<CellSet>();

    CellList _prev_diff;

    bool next_state(const CellSet &board, Cell param) const;

    int _iteration = 0;

    const int _threads;

    int adjacent_alive(const CellSet &board, Cell coords) const;

    CellList build_diff(const CellSet &board) const;
};

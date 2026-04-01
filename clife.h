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
#include <condition_variable>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>
#include <vstd.h>


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
    using CellList = std::vector<Cell>;
    using IndexList = std::vector<int>;

    struct FrameView {
        const std::vector<std::uint8_t> *cells = nullptr;
        const IndexList *changed = nullptr;
        int view_width = 0;
        int view_height = 0;
        int width = 0;
        int height = 0;
        int stride = 0;
        int origin_x = 0;
        int origin_y = 0;
        bool full_refresh = false;
    };

    LifeBoard(std::shared_ptr<CellSet> board, int threads, int width = 0, int height = 0);

    ~LifeBoard();

    FrameView iterate();

    CellSet snapshot() const;

    bool alive(Cell cell) const;

    int width() const;

    int height() const;

private:
    struct RowRange {
        int begin = 0;
        int end = 0;
    };

    void evaluate_chunk(std::size_t chunk_index);

    void worker_loop(std::size_t chunk_index);

    void build_neighbor_counts();

    void apply_neighbor_deltas();

    bool in_bounds(Cell cell) const;

    int to_index(Cell cell) const;

    Cell to_cell(int index) const;

    int _width = 0;
    int _height = 0;
    int _stride = 0;
    int _view_width = 0;
    int _view_height = 0;
    int _origin_x = 0;
    int _origin_y = 0;
    int _participants = 1;
    int _live_cells = 0;
    bool _first_frame = true;

    std::vector<std::uint8_t> _front;
    std::vector<std::uint8_t> _back;
    std::vector<std::uint8_t> _neighbor_counts;
    std::array<int, 8> _neighbor_offsets = {};
    IndexList _changed_indices;
    std::vector<RowRange> _chunk_ranges;
    std::vector<IndexList> _chunk_changes;
    std::vector<std::thread> _workers;

    std::mutex _worker_mutex;
    std::condition_variable _worker_cv;
    std::condition_variable _done_cv;
    std::size_t _job_generation = 0;
    std::size_t _completed_workers = 0;
    bool _stop_workers = false;
};

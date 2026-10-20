# Vendored from NVIDIA Fold-CP (MIT): https://github.com/NVIDIA-BioNeMo/boltz-cp @ f76f37a77c854b56b6250b426af8c2d63b501f7f
# Source: src/boltz/distributed/comm.py

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


from typing import Any

import torch
import torch.distributed as dist

from bioemu2.context_parallel.distributed.utils import (
    LayoutMap,
    get_group_rank_from_axial_shift,
)


class One2OneComm:
    def __init__(
        self,
        group: dist.ProcessGroup,
        rank_send_to: int,
        rank_recv_from: int,
        parity: bool | None = None,
    ):
        """
        Initializes a One2OneComm instance for point-to-point communication.

        Arguments:
            group (dist.ProcessGroup): The process group that provides the communication.
            rank_send_to (int): The rank within the group to send data to.
            rank_recv_from (int): The rank within the group to receive data from.
            parity (bool): If True, issue [isend, irecv]; otherwise issue [irecv, isend]
                in batch_isend_irecv. If None, parity is `rank % 2`, where `rank` is the
                calling rank's index in the WORLD group. If self.is_self_comm is True, i.e.,
                `rank_send_to == rank and rank_recv_from == rank`, this argument has no effect.
                The motivation of setting the parity is to avoid potential deadlocks in NCCL backend
                when doing batch_isend_irecv.

        Note: rank_send_to and rank_recv_from must be ranks within the input process group.

        Raises:
            ValueError: If rank_send_to or rank_recv_from is not a valid rank within the group.
        """
        self.group = group

        self.rank = dist.get_rank(self.group)
        self.world_size = dist.get_world_size(self.group)

        if rank_send_to >= self.world_size:
            raise ValueError(f"rank_send_to >= world_size {self.world_size}")
        if rank_recv_from >= self.world_size:
            raise ValueError(f"rank_recv_from >= world_size {self.world_size}")
        # make all comm functions no-ops if self-send and self-recv
        is_self_send = rank_send_to == self.rank
        is_self_recv = rank_recv_from == self.rank
        if is_self_send != is_self_recv:
            raise ValueError(
                "Asymmetric send/recv tends to cause NCCL backend deadlocking "
                f"and it's not supported: is_self_send: {is_self_send}, "
                f"is_self_recv: {is_self_recv}"
            )
        self.is_self_comm = is_self_send
        self._rank_in_group_send_to = rank_send_to
        self._rank_in_group_recv_from = rank_recv_from

        self.parity = parity

        if not self.is_self_comm:
            # convert to global rank
            self.rank_send_to = dist.get_global_rank(self.group, rank_send_to)
            self.rank_recv_from = dist.get_global_rank(self.group, rank_recv_from)

            if self.parity is None:
                self.parity = self.rank % 2

            self._queue_send_recv: list[Any] = []
            self._work_to_finish = None

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the One2OneComm instance.

        This method enables the One2OneComm object to be deep copied using the copy.deepcopy() function.
        It creates a new One2OneComm instance with the same communication parameters as the original.

        Args:
            memo (dict): Dictionary used by deepcopy to avoid circular references.

        Returns:
            One2OneComm: A new One2OneComm instance with identical configuration to the original.
        """
        return One2OneComm(
            self.group,
            self._rank_in_group_send_to,
            self._rank_in_group_recv_from,
            self.parity,
        )

    def _prep_batch_isend_irecv(
        self,
        to_send: torch.Tensor,
        to_recv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Prepare tensors and communication operations for batch send/receive.

        This private method sets up the tensor operations and queues the communication
        operations for later dispatch. It handles both self-communication (where send
        and receive are on the same rank) and inter-rank communication.

        Args:
            to_send (torch.Tensor): The tensor to be sent to the target rank.
            to_recv (Optional[torch.Tensor], optional): The tensor buffer to receive data into.
                If None, a new tensor with the same shape and properties as `to_send` will be created.

        Returns:
            torch.Tensor: The tensor that will contain the received data. For self-communication,
                this is either a clone of `to_send` or `to_recv` with data copied from `to_send`.
                For inter-rank communication, this is the buffer where received data will be stored.

        Note:
            - For self-communication (`is_self_comm=True`), the data is immediately copied
              and no communication operations are queued.
            - For inter-rank communication, P2P operations are queued based on parity to
              avoid potential deadlocks in NCCL backend.
            - The order of send/receive operations depends on the parity flag to ensure
              consistent ordering across ranks.
        """
        if self.is_self_comm:
            # the copy semantics remain even if self.is_self_comm
            if to_recv is None:
                ans = to_send.detach().clone()
            else:
                ans = to_recv
                ans.copy_(to_send)
            return ans

        ans = torch.empty_like(to_send) if to_recv is None else to_recv

        if self.parity:
            send_op = dist.P2POp(
                dist.isend,
                to_send,
                self.rank_send_to,
                group=self.group,
            )
            recv_op = dist.P2POp(
                dist.irecv,
                ans,
                self.rank_recv_from,
                group=self.group,
            )
            self._queue_send_recv.append(send_op)
            self._queue_send_recv.append(recv_op)
        else:
            recv_op = dist.P2POp(
                dist.irecv,
                ans,
                self.rank_recv_from,
                group=self.group,
            )
            send_op = dist.P2POp(
                dist.isend,
                to_send,
                self.rank_send_to,
                group=self.group,
            )
            self._queue_send_recv.append(recv_op)
            self._queue_send_recv.append(send_op)
        return ans

    def _dispatch(self):
        """
        Dispatch all queued communication operations.

        This private method initiates all point-to-point communication operations that have been
        queued by previous calls to `_prep_batch_isend_irecv`. The operations are dispatched
        asynchronously using `dist.batch_isend_irecv`.

        Raises:
            RuntimeError: If there are already unfinished communications in the queue when trying
                to dispatch new operations. This prevents overlapping communication operations
                which could lead to undefined behavior.

        Note:
            - For self-communication (`is_self_comm=True`), this method does nothing as no
              actual network communication is required.
            - After dispatching, the work handles are stored in `_work_to_finish` for later
              synchronization via `wait_until_finished()`.
            - This method should only be called after communication operations have been
              queued via `_prep_batch_isend_irecv()`.
        """
        if self.is_self_comm:
            return
        if self._work_to_finish is not None:
            raise RuntimeError(
                "There is unfinished communication in queue. Cannot dispatch new communication"
            )
        self._work_to_finish = dist.batch_isend_irecv(self._queue_send_recv)

    def wait_until_finished(self):
        """
        Wait for all dispatched communication operations to complete.

        This method blocks until all previously dispatched communication operations have
        finished. It ensures data consistency by synchronizing all pending send/receive
        operations before proceeding.

        Raises:
            RuntimeError: If called when there are no unfinished communications in the queue.
                This typically happens when `wait_until_finished()` is called without a
                preceding `_dispatch()` call.

        Note:
            - For self-communication (`is_self_comm=True`), this method returns immediately
              as no actual network communication needs to be synchronized.
            - After completion, the internal communication queue and work handles are reset,
              allowing new communication operations to be queued.
            - This method must be called after `_dispatch()` to ensure communication
              operations have completed before accessing the received data.
        """
        if self.is_self_comm:
            return
        if self._work_to_finish is None:
            raise RuntimeError("Cannot wait without unfinished communication in queue")
        for work in self._work_to_finish:
            work.wait()
        self._work_to_finish = None
        self._queue_send_recv = []

    def enqueue_to_dispatch(
        self,
        to_send: torch.Tensor,
        to_recv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Enqueue a communication operation and immediately dispatch it.

        This method combines the functionality of `_prep_batch_isend_irecv()` and `_dispatch()`
        in a single call. It prepares the tensors for communication, queues the operations,
        and immediately dispatches them for execution.

        Args:
            to_send (torch.Tensor): The tensor to be sent to the target rank.
            to_recv (Optional[torch.Tensor], optional): The tensor buffer to receive data into.
                If None, a new tensor with the same shape and properties as `to_send` will be created.

        Returns:
            torch.Tensor: The tensor that will contain the received data. For self-communication,
                this contains the copied data immediately. For inter-rank communication, this is
                the buffer where data will be received once the communication completes.

        Note:
            - For self-communication (`is_self_comm=True`), the data is immediately available
              in the returned tensor.
            - For inter-rank communication, you must call `wait_until_finished()` before
              accessing the data in the returned tensor to ensure the communication has completed.
            - This is a convenience method that internally calls `_prep_batch_isend_irecv()`
              followed by `_dispatch()`.

        Example:
            ```python
            comm = One2OneComm(group, send_rank, recv_rank)
            recv_tensor = comm.enqueue_to_dispatch(send_tensor)
            comm.wait_until_finished()  # Wait for completion before using recv_tensor
            ```
        """
        recv = self._prep_batch_isend_irecv(to_send, to_recv)
        if self.is_self_comm:
            return recv
        self._dispatch()
        return recv


class TransposeComm(One2OneComm):
    def __init__(self, process_group: dist.ProcessGroup, group_layout: LayoutMap):
        if group_layout.shape is None:
            raise ValueError("group_layout must have a shape")

        self.world_size = dist.get_world_size(process_group)
        if self.world_size != group_layout.numel:
            raise ValueError("Inconsistent world_size with the num elements of group_layout")

        if len(group_layout.shape) != 2:
            raise ValueError(f"{self.__class__} only supports 2D group layout")

        if group_layout.shape[0] != group_layout.shape[1]:
            raise ValueError(f"group_layout.shape {group_layout.shape} is not square")

        self.group_layout = group_layout

        self.global_rank = dist.get_rank()
        self.group_rank = dist.get_rank(process_group)
        self.rank_coords: tuple[int, ...] = self.group_layout.unravel(self.group_rank)

        transpose_group_rank = self.group_layout(self.rank_coords[::-1])
        self.transpose_rank = dist.get_global_rank(process_group, transpose_group_rank)

        self.parity_transpose = self.rank_coords[0] < self.rank_coords[1]

        # Call One2OneComm's __init__ instead of creating a separate comm instance
        super().__init__(
            process_group,
            transpose_group_rank,
            transpose_group_rank,
            parity=self.parity_transpose,
        )

    def __deepcopy__(self, memo):
        return TransposeComm(self.group, self.group_layout)


def ternary_parity(my_rank: int, send_rank: int, recv_rank: int) -> bool:
    """
    Determines parity for communication ordering based on rank relationships.

    Used to establish consistent communication ordering between three ranks to avoid deadlocks.
    Returns True if the current rank is less than both the send and receive ranks.

    Args:
        my_rank: Current process rank
        send_rank: Rank to send data to
        recv_rank: Rank to receive data from

    Returns:
        bool: True if current rank is less than both send and receive ranks, False otherwise
    """
    return my_rank < min(send_rank, recv_rank)


class Ring2DComm:
    """
    Implements communication primitives for distributed operations on a 2D grid of devices.

    This class provides general-purpose ring communication patterns for operations like
    TriangleMultiplication and OuterProductMean across a 2D grid of devices.

    The communication patterns implemented include:
    1. Transpose communication for matrix operations
    2. Row-wise ring communication (left shifts)
    3. Column-wise ring communication (up shifts)

    Parameters
    ----------
    group_2d : dist.ProcessGroup
        The process group representing the 2D grid of devices. This should include
        all processes in the distributed computation.
    group_col : dist.ProcessGroup
        A subprocess group that provides communication between ranks in the same column.
    group_layout : LayoutMap
        A mapping from the 2D grid index to the flattened index of the devices on the 2D grid.
        Must represent a square grid (same dimensions in both axes).

    Notes
    -----
    The class implements various communication patterns needed for distributed matrix
    operations, including initial communication (with different shift patterns based on
    coordinates) and subsequent iterations (with fixed shifts).

    Communication ordering is carefully managed to prevent deadlocks by using
    ternary_parity to determine consistent send/receive ordering across different ranks.
    """

    def __init__(
        self,
        group_2d: dist.ProcessGroup,
        group_col: dist.ProcessGroup,
        group_layout: LayoutMap,
    ):
        """
        Ring comm over a 2d grid of devices with comm happening along both axes
        Arguments:
            group_2d: Group torch process group that provides communication
                across the full cross-device
            group_col: Subprocess group that provides communication
                between ranks in the same column
            group_layout: mapping from the 2d grid index to the flatten index
            of the devices on the 2d grid
        """
        self.group_2d = group_2d
        self.group_col = group_col
        self.group_layout = group_layout
        ranks_group_2d = set(dist.get_process_group_ranks(self.group_2d))
        ranks_group_col = set(dist.get_process_group_ranks(self.group_col))

        if not ranks_group_col.issubset(ranks_group_2d):
            raise ValueError("The col ranks are not a subset of ranks_group_2d")

        self.size_2d = dist.get_world_size(self.group_2d)

        if self.size_2d != self.group_layout.numel:
            raise ValueError(
                f"size of group_2d {self.size_2d} differs from the number of elements in group_layout {self.group_layout.numel}"
            )

        if self.group_layout.shape[0] != self.group_layout.shape[1]:
            raise ValueError(f"group_layout.shape {self.group_layout.shape} is not square")

        self.rank_2d = dist.get_rank(self.group_2d)
        self.coord_2d = self.group_layout.unravel(self.rank_2d)

        # all the send/recv ranks must be global in order to use isend/irecv
        # only need transpose at the beginning of the batched GEMM for b or a
        self.comm_2d_trans = TransposeComm(self.group_2d, self.group_layout)

        # always do left shift per row
        # for iteration 0, i'th row left shift by i column
        self.send_rank_row_init = get_group_rank_from_axial_shift(
            self.coord_2d, 1, -self.coord_2d[0], self.group_layout
        )
        self.recv_rank_row_init = get_group_rank_from_axial_shift(
            self.coord_2d, 1, self.coord_2d[0], self.group_layout
        )

        self.comm_row_init = One2OneComm(
            self.group_2d,
            self.send_rank_row_init,
            self.recv_rank_row_init,
            parity=ternary_parity(self.rank_2d, self.send_rank_row_init, self.recv_rank_row_init),
        )
        # for other iterations left shift by 1 col
        self.send_rank_row = get_group_rank_from_axial_shift(
            self.coord_2d, 1, -1, self.group_layout
        )
        self.recv_rank_row = get_group_rank_from_axial_shift(self.coord_2d, 1, 1, self.group_layout)

        self.comm_row = One2OneComm(
            self.group_2d,
            self.send_rank_row,
            self.recv_rank_row,
            parity=ternary_parity(self.rank_2d, self.send_rank_row, self.recv_rank_row),
        )

        # always do up shift per col
        # for iteration 0, j'th col up shift by j row
        self.send_rank_col_init = get_group_rank_from_axial_shift(
            self.coord_2d, 0, -self.coord_2d[1], self.group_layout
        )
        self.recv_rank_col_init = get_group_rank_from_axial_shift(
            self.coord_2d, 0, self.coord_2d[1], self.group_layout
        )
        self.comm_col_init = One2OneComm(
            self.group_2d,
            self.send_rank_col_init,
            self.recv_rank_col_init,
            parity=ternary_parity(self.rank_2d, self.send_rank_col_init, self.recv_rank_col_init),
        )
        # for other iterations, up shift by 1 row
        self.send_rank_col = get_group_rank_from_axial_shift(
            self.coord_2d, 0, -1, self.group_layout
        )
        self.recv_rank_col = get_group_rank_from_axial_shift(self.coord_2d, 0, 1, self.group_layout)
        self.comm_col = One2OneComm(
            self.group_2d,
            self.send_rank_col,
            self.recv_rank_col,
            parity=ternary_parity(self.rank_2d, self.send_rank_col, self.recv_rank_col),
        )

        # fused communication for transposition and initial row/col shift in backward
        coords_transpose = self.coord_2d[::-1]
        self.send_rank_transpose_row_init = get_group_rank_from_axial_shift(
            coords_transpose, 1, -coords_transpose[0], self.group_layout
        )  # shifting the transposed rank
        recv_rank_transpose_row_init = get_group_rank_from_axial_shift(
            self.coord_2d, 1, self.coord_2d[0], self.group_layout
        )  # counter-shifting
        self.recv_rank_transpose_row_init = self.group_layout(
            self.group_layout.unravel(recv_rank_transpose_row_init)[::-1]
        )  # counter-transposition
        self.comm_transpose_row_init = One2OneComm(
            self.group_2d,
            self.send_rank_transpose_row_init,
            self.recv_rank_transpose_row_init,
            parity=ternary_parity(
                self.rank_2d,
                self.send_rank_transpose_row_init,
                self.recv_rank_transpose_row_init,
            ),
        )

        self.send_rank_transpose_col_init = get_group_rank_from_axial_shift(
            coords_transpose, 0, -coords_transpose[1], self.group_layout
        )  # shifting the transposed rank
        recv_rank_transpose_col_init = get_group_rank_from_axial_shift(
            self.coord_2d, 0, self.coord_2d[1], self.group_layout
        )  # counter-shifting
        self.recv_rank_transpose_col_init = self.group_layout(
            self.group_layout.unravel(recv_rank_transpose_col_init)[::-1]
        )  # counter-transposition
        self.comm_transpose_col_init = One2OneComm(
            self.group_2d,
            self.send_rank_transpose_col_init,
            self.recv_rank_transpose_col_init,
            parity=ternary_parity(
                self.rank_2d,
                self.send_rank_transpose_col_init,
                self.recv_rank_transpose_col_init,
            ),
        )


class AttentionPairBiasComm:
    def __init__(
        self,
        process_group: dist.ProcessGroup,
        group_layout: LayoutMap,
        cp_axis_0_group: dist.ProcessGroup,
        cp_axis_1_group: dist.ProcessGroup,
    ):
        self.process_group = process_group
        self.cp_axis_0_group = cp_axis_0_group
        self.cp_axis_1_group = cp_axis_1_group

        if group_layout.shape is None:
            raise ValueError("group_layout must have a shape")

        self.world_size = dist.get_world_size(self.process_group)
        if self.world_size != group_layout.numel:
            raise ValueError("Inconsistent world_size with the num elements of group_layout")

        if len(group_layout.shape) != 2:
            raise ValueError(f"{self.__class__} only supports 2D group layout")

        if group_layout.shape[0] != group_layout.shape[1]:
            raise ValueError(f"group_layout.shape {group_layout.shape} is not square")

        self.group_layout = group_layout

        self.global_rank = dist.get_rank()
        self.group_rank = dist.get_rank(self.process_group)
        self.rank_coords: tuple[int, ...] = self.group_layout.unravel(self.group_rank)

        self.comm_transpose_k = TransposeComm(
            self.process_group, self.group_layout
        )  # also used for backward
        self.comm_transpose_v = TransposeComm(
            self.process_group, self.group_layout
        )  # also used for backward
        self.comm_transpose_mask = TransposeComm(self.process_group, self.group_layout)

        # for k, v and z comm
        self.send_rank_kvz = get_group_rank_from_axial_shift(
            self.rank_coords, 1, 1, self.group_layout
        )
        self.recv_rank_kvz = get_group_rank_from_axial_shift(
            self.rank_coords, 1, -1, self.group_layout
        )
        self.parity = self.rank_coords[1] % 2 == 1
        self.comm_k = One2OneComm(
            self.process_group,
            self.send_rank_kvz,
            self.recv_rank_kvz,
            parity=self.parity,
        )
        self.comm_v = One2OneComm(
            self.process_group,
            self.send_rank_kvz,
            self.recv_rank_kvz,
            parity=self.parity,
        )
        self.comm_z = One2OneComm(
            self.process_group,
            self.send_rank_kvz,
            self.recv_rank_kvz,
            parity=self.parity,
        )

    def __deepcopy__(self, memo):
        return AttentionPairBiasComm(
            self.process_group,
            self.group_layout,
            self.cp_axis_0_group,
            self.cp_axis_1_group,
        )

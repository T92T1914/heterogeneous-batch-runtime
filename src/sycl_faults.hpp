#pragma once
// Only compiled into the separate fault-test executable, never the installed library.
namespace hbr::test {
void fail_at(int checkpoint);
// The injected failure is a sycl::exception instead of a host exception.
void fail_as_sycl(bool);
// Completion finds a kept asynchronous SYCL error, as the queue's handler would leave it.
void fail_async(bool);
int checkpoints();
int live_allocations();
int drains();
int drain_errors();
}

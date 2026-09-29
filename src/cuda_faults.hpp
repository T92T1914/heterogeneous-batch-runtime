#pragma once
// Only compiled into the separate fault-test executable, never the installed library.
namespace hbr::test {
void fail_at(int checkpoint);
int checkpoints();
int live_resources();
int drains();
int drain_errors();
}

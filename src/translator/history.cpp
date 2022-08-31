#include "history.h"

namespace marian {

History::History(size_t lineNo, Word eosId, float alpha, float wp)
    : lineNo_(lineNo), eosId_(eosId), alpha_(alpha), wp_(wp) {}
}  // namespace marian

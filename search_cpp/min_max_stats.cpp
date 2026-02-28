#include "min_max_stats.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace rainbow::search {

MinMaxStats::MinMaxStats(
    const std::vector<double>& known_bounds,
    const bool soft_update,
    const double min_max_epsilon)
    : soft_update_(soft_update),
      min_max_epsilon_(min_max_epsilon),
      max_(known_bounds.size() >= 2 ? known_bounds[1] : -std::numeric_limits<double>::infinity()),
      min_(known_bounds.size() >= 2 ? known_bounds[0] : std::numeric_limits<double>::infinity()) {
    if (known_bounds.size() == 1 || known_bounds.size() > 2) {
        throw std::invalid_argument("known_bounds must contain either 0 or 2 values.");
    }
}

void MinMaxStats::update(const double value) {
    max_ = std::max(max_, value);
    min_ = std::min(min_, value);
}

double MinMaxStats::normalize(const double value) const {
    if (max_ > min_) {
        double denom = max_ - min_;
        if (soft_update_) {
            denom = std::max(denom, min_max_epsilon_);
        }
        return (value - min_) / denom;
    }
    return value;
}

double MinMaxStats::min() const {
    return min_;
}

double MinMaxStats::max() const {
    return max_;
}

bool MinMaxStats::soft_update() const {
    return soft_update_;
}

double MinMaxStats::min_max_epsilon() const {
    return min_max_epsilon_;
}

std::string MinMaxStats::repr() const {
    std::ostringstream out;
    out << "min: " << min_ << ", max: " << max_;
    return out.str();
}

}  // namespace rainbow::search

#pragma once

#include <string>
#include <vector>

namespace rainbow::search {

class MinMaxStats {
public:
    explicit MinMaxStats(
        const std::vector<double>& known_bounds = {},
        bool soft_update = false,
        double min_max_epsilon = 0.01);

    void update(double value);
    double normalize(double value) const;

    double min() const;
    double max() const;
    bool soft_update() const;
    double min_max_epsilon() const;

    std::string repr() const;

private:
    bool soft_update_;
    double min_max_epsilon_;
    double max_;
    double min_;
};

}  // namespace rainbow::search

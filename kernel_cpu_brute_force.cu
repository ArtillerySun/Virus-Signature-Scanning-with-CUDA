#include "kseq/kseq.h"
#include "common.h"

const int P = 97;

inline int get_phred(const char &c) {
    return c - 33;
}

inline bool check(const char &e, const char &f) {
    return e == 'N' || f == 'N' || e == f;
}

void solve(const int hash, const klibpp::KSeq& sample, const klibpp::KSeq& signature, std::vector<MatchResult>& matches) {
    int len = signature.seq.length();
    double mx_score = 0;
    bool is_find = false;
    for (int i = 0; i + len - 1 < sample.seq.length(); i++) {
        double sum = 0;
        bool flag = true;
        for (int j = 0; j < len; j++) {
            if (!check(sample.seq[i + j], signature.seq[j])) {
                flag = false;
                break;
            }
            sum += get_phred(sample.qual[i + j]);
        }

        if (flag) {
            is_find = true;
            mx_score = std::max(mx_score, sum / len);
        }

        sum = 0;
    }

    if (is_find) {
        matches.push_back({sample.name, signature.name, mx_score, hash});
    }
}

void runMatcher(const std::vector<klibpp::KSeq>& samples, const std::vector<klibpp::KSeq>& signatures, std::vector<MatchResult>& matches) {

    for (auto& sample : samples) {
        int hash = 0;
        for (auto c : sample.qual) {
            hash = (hash + get_phred(c)) % P;
        }
        for (auto& signature : signatures) {
            solve(hash, sample, signature, matches);
        }
    }
}

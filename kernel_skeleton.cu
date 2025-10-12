#include "kseq/kseq.h"
#include "common.h"

const int P = 97;

void solve(const int hash, const klibpp::KSeq& sample, const klibpp::KSeq& signature, std::vector<MatchResult>& matches) {
    int len = signature.seq.length();
    double mx_score = 0;
    bool is_find = false;
    for (int i = 0; i + len - 1 < sample.seq.length(); i++) {
        double sum = 0;
        bool flag = true;
        for (int j = 0; j < len; j++) {
            if (sample.seq[i + j] != 'N' && signature.seq[j] != 'N' && 
                sample.seq[i + j] != signature.seq[j]) {
                flag = false;
                break;
            }
            sum += sample.qual[i + j] == 'N' ? 0 : sample.qual[i + j] - 33;
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
            int a = c == 'N' ? 0 : c - 33;
            hash = (hash + a) % P;
        }
        for (auto& signature : signatures) {
            solve(hash, sample, signature, matches);
        }
    }
}

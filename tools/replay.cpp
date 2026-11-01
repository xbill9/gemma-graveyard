// replay.cpp -- feed recorded answers back through a GGUF and report, at every token of the answer, the
// model's top two next tokens, the logit gap between them, and the rank of the token that was recorded.
//
//   replay <model.gguf> <in.tsv> <out.jsonl> [threads]
//
// in.tsv: one line per answer, key<TAB>hex(prompt)<TAB>hex(answer). Prompt is tokenized with special tokens
// parsed, the answer as plain text. Built against libllama from a llama.cpp checkout; see tools/build_replay.sh.
#include "llama.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::string unhex(const std::string &h) {
    std::string o;
    o.reserve(h.size() / 2);
    for (size_t i = 0; i + 1 < h.size(); i += 2) o.push_back((char)std::stoi(h.substr(i, 2), nullptr, 16));
    return o;
}
static std::vector<llama_token> tok(const llama_vocab *v, const std::string &s, bool special) {  // special: BOS added, special tokens parsed
    int n = -llama_tokenize(v, s.c_str(), s.size(), nullptr, 0, special, special);
    std::vector<llama_token> t(n);
    llama_tokenize(v, s.c_str(), s.size(), t.data(), n, special, special);
    return t;
}
static std::string esc(const std::string &s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c < 0x20) { char b[8]; snprintf(b, 8, "\\u%04x", c); o += b; }
        else o += c;
    }
    return o;
}
static std::string piece(const llama_vocab *v, llama_token t) {
    char b[256];
    int n = llama_token_to_piece(v, t, b, sizeof b, 0, true);
    return n > 0 ? std::string(b, n) : "";
}

int main(int argc, char **argv) {
    if (argc < 4) return 2;
    int threads = argc > 4 ? atoi(argv[4]) : 8;
    llama_backend_init();
    llama_log_set([](ggml_log_level l, const char *t, void *) { if (l >= GGML_LOG_LEVEL_ERROR) fputs(t, stderr); }, nullptr);
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model *m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 3;
    const llama_vocab *v = llama_model_get_vocab(m);
    const int nv = llama_vocab_n_tokens(v);
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 4096; cp.n_ubatch = 64; cp.n_threads = threads; cp.n_threads_batch = threads;
    llama_context *c = llama_init_from_model(m, cp);
    std::ifstream in(argv[2]);
    std::ofstream out(argv[3], std::ios::app);
    std::string line;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string key, hp, ha;
        std::getline(ss, key, '\t'); std::getline(ss, hp, '\t'); std::getline(ss, ha, '\t');
        auto p = tok(v, unhex(hp), true);
        auto a = tok(v, unhex(ha), false);
        std::vector<llama_token> all = p;
        all.insert(all.end(), a.begin(), a.end());
        if ((int)all.size() > 4000) all.resize(4000);
        llama_memory_clear(llama_get_memory(c), true);
        const int np = p.size(), n = all.size();
        const int step = 64;
        std::ostringstream o;
        o << "{\"key\":\"" << key << "\",\"prompt_tokens\":" << np << ",\"t\":[";
        bool first = true;
        for (int s = 0; s < n; s += step) {
            int e = std::min(n, s + step);
            llama_batch b = llama_batch_init(e - s, 0, 1);
            for (int i = s; i < e; i++) {
                int k = i - s;
                b.token[k] = all[i]; b.pos[k] = i; b.n_seq_id[k] = 1; b.seq_id[k][0] = 0;
                b.logits[k] = (i >= np - 1) && (i + 1 < n);
            }
            b.n_tokens = e - s;
            if (llama_decode(c, b)) { std::cerr << "decode failed " << key << "\n"; return 4; }
            for (int i = s; i < e; i++) {
                if (!b.logits[i - s]) continue;
                const float *l = llama_get_logits_ith(c, i - s);
                int t1 = 0, t2 = -1;
                for (int j = 1; j < nv; j++) {
                    if (l[j] > l[t1]) { t2 = t1; t1 = j; }
                    else if (t2 < 0 || l[j] > l[t2]) t2 = j;
                }
                llama_token r = all[i + 1];
                int rank = 1;
                for (int j = 0; j < nv; j++) rank += l[j] > l[r];
                if (!first) o << ",";
                first = false;
                o << "[\"" << esc(piece(v, r)) << "\"," << rank << "," << l[t1] - l[r] << ",\"" << esc(piece(v, t1)) << "\",\""
                  << esc(piece(v, t2)) << "\"," << l[t1] - l[t2] << "]";
            }
            llama_batch_free(b);
        }
        o << "]}\n";
        out << o.str();
        out.flush();
    }
    llama_free(c); llama_model_free(m); llama_backend_free();
    return 0;
}

/* Minimal OpenAI-compatible endpoint for the Kimi K3 driver (2026-11-01): one request at a time, greedy
 * decoding through ds4_engine_generate_argmax. GET /v1/models; POST /v1/chat/completions with or without
 * "stream". Prompts are rendered with the K3 chat format (deltafin chat.rs rules, built here): system/user/
 * assistant/tool messages, tools + tool_choice (tool calls come back as tool_calls), response_format, thinking. */
#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "ds4.h"

/* ------------------------------------------------------------------ growable string */
typedef struct { char *p; size_t len, cap; } sbuf;
static void sb_reserve(sbuf *b, size_t extra) {
    if (b->len + extra + 1 <= b->cap) return;
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->len + extra + 1) cap *= 2;
    b->p = realloc(b->p, cap);
    if (!b->p) { fprintf(stderr, "ds4: K3 serve: out of memory\n"); exit(1); }
    b->cap = cap;
}
static void sb_putn(sbuf *b, const char *s, size_t n) { sb_reserve(b, n); memcpy(b->p + b->len, s, n); b->len += n; b->p[b->len] = 0; }
static void sb_puts(sbuf *b, const char *s) { sb_putn(b, s, strlen(s)); }
static void sb_putc(sbuf *b, char c) { sb_putn(b, &c, 1); }
static void sb_free(sbuf *b) { free(b->p); b->p = NULL; b->len = b->cap = 0; }
static void sb_put_utf8(sbuf *b, unsigned cp) {
    char u[4]; int n = 0;
    if (cp < 0x80) { u[n++] = (char)cp; }
    else if (cp < 0x800) { u[n++] = (char)(0xC0 | (cp >> 6)); u[n++] = (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { u[n++] = (char)(0xE0 | (cp >> 12)); u[n++] = (char)(0x80 | ((cp >> 6) & 0x3F)); u[n++] = (char)(0x80 | (cp & 0x3F)); }
    else { u[n++] = (char)(0xF0 | (cp >> 18)); u[n++] = (char)(0x80 | ((cp >> 12) & 0x3F)); u[n++] = (char)(0x80 | ((cp >> 6) & 0x3F)); u[n++] = (char)(0x80 | (cp & 0x3F)); }
    sb_putn(b, u, (size_t)n);
}
static void sb_put_json_string(sbuf *b, const char *s, size_t n) {
    sb_putc(b, '"');
    for (size_t i = 0; i < n; i++) {
        const unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"': sb_puts(b, "\\\""); break;
        case '\\': sb_puts(b, "\\\\"); break;
        case '\n': sb_puts(b, "\\n"); break;
        case '\r': sb_puts(b, "\\r"); break;
        case '\t': sb_puts(b, "\\t"); break;
        default:
            if (c < 0x20) { char esc[8]; snprintf(esc, sizeof(esc), "\\u%04x", c); sb_puts(b, esc); }
            else sb_putc(b, (char)c);
        }
    }
    sb_putc(b, '"');
}

/* ------------------------------------------------------------------ tiny JSON parser */
typedef enum { JV_NULL, JV_BOOL, JV_NUM, JV_STR, JV_ARR, JV_OBJ } jv_type;
typedef struct jv {
    jv_type type;
    double num;
    int boolean;
    char *str;
    struct jv **items;
    char **keys;
    int n;
} jv;
static void jv_free(jv *v) {
    if (!v) return;
    free(v->str);
    for (int i = 0; i < v->n; i++) { jv_free(v->items[i]); if (v->keys) free(v->keys[i]); }
    free(v->items); free(v->keys); free(v);
}
static const char *jp_ws(const char *p) { while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++; return p; }
static int jp_hex4(const char *p, unsigned *out) {
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        const char c = p[i]; v <<= 4;
        if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
        else return 0;
    }
    *out = v; return 1;
}
static char *jp_string(const char **pp) {
    const char *p = *pp;
    if (*p != '"') return NULL;
    p++;
    sbuf b = {0};
    sb_reserve(&b, 16);
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            const char e = *p++;
            unsigned cp = 0;
            switch (e) {
            case '"': sb_putc(&b, '"'); break;
            case '\\': sb_putc(&b, '\\'); break;
            case '/': sb_putc(&b, '/'); break;
            case 'b': sb_putc(&b, '\b'); break;
            case 'f': sb_putc(&b, '\f'); break;
            case 'n': sb_putc(&b, '\n'); break;
            case 'r': sb_putc(&b, '\r'); break;
            case 't': sb_putc(&b, '\t'); break;
            case 'u':
                if (!jp_hex4(p, &cp)) { sb_free(&b); return NULL; }
                p += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF && p[0] == '\\' && p[1] == 'u') {
                    unsigned lo = 0;
                    if (jp_hex4(p + 2, &lo) && lo >= 0xDC00 && lo <= 0xDFFF) { cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u); p += 6; }
                }
                sb_put_utf8(&b, cp);
                break;
            default: sb_free(&b); return NULL;
            }
        } else {
            sb_putc(&b, *p++);
        }
    }
    if (*p != '"') { sb_free(&b); return NULL; }
    *pp = p + 1;
    return b.p;
}
static jv *jp_value(const char **pp, int depth);
static jv *jv_new(jv_type t) { jv *v = calloc(1, sizeof(jv)); if (!v) exit(1); v->type = t; return v; }
static void jv_push(jv *v, char *key, jv *item) {
    v->items = realloc(v->items, sizeof(jv *) * (size_t)(v->n + 1));
    if (v->type == JV_OBJ) v->keys = realloc(v->keys, sizeof(char *) * (size_t)(v->n + 1));
    if (!v->items || (v->type == JV_OBJ && !v->keys)) exit(1);
    v->items[v->n] = item;
    if (v->type == JV_OBJ) v->keys[v->n] = key;
    v->n++;
}
static jv *jp_value(const char **pp, int depth) {
    if (depth > 64) return NULL;
    const char *p = jp_ws(*pp);
    jv *v = NULL;
    if (*p == '{') {
        v = jv_new(JV_OBJ); p = jp_ws(p + 1);
        if (*p == '}') { *pp = p + 1; return v; }
        for (;;) {
            p = jp_ws(p);
            char *key = jp_string(&p);
            if (!key) { jv_free(v); return NULL; }
            p = jp_ws(p);
            if (*p != ':') { free(key); jv_free(v); return NULL; }
            p++;
            jv *item = jp_value(&p, depth + 1);
            if (!item) { free(key); jv_free(v); return NULL; }
            jv_push(v, key, item);
            p = jp_ws(p);
            if (*p == ',') { p++; continue; }
            if (*p == '}') { p++; break; }
            jv_free(v); return NULL;
        }
    } else if (*p == '[') {
        v = jv_new(JV_ARR); p = jp_ws(p + 1);
        if (*p == ']') { *pp = p + 1; return v; }
        for (;;) {
            jv *item = jp_value(&p, depth + 1);
            if (!item) { jv_free(v); return NULL; }
            jv_push(v, NULL, item);
            p = jp_ws(p);
            if (*p == ',') { p++; continue; }
            if (*p == ']') { p++; break; }
            jv_free(v); return NULL;
        }
    } else if (*p == '"') {
        char *s = jp_string(&p);
        if (!s) return NULL;
        v = jv_new(JV_STR); v->str = s;
    } else if (!strncmp(p, "true", 4)) { v = jv_new(JV_BOOL); v->boolean = 1; p += 4; }
    else if (!strncmp(p, "false", 5)) { v = jv_new(JV_BOOL); v->boolean = 0; p += 5; }
    else if (!strncmp(p, "null", 4)) { v = jv_new(JV_NULL); p += 4; }
    else if (*p == '-' || (*p >= '0' && *p <= '9')) {
        char *end = NULL;
        const double d = strtod(p, &end);
        if (end == p) return NULL;
        v = jv_new(JV_NUM); v->num = d; p = end;
    } else {
        return NULL;
    }
    *pp = p;
    return v;
}
static jv *jv_get(const jv *obj, const char *key) {
    if (!obj || obj->type != JV_OBJ) return NULL;
    for (int i = 0; i < obj->n; i++) if (!strcmp(obj->keys[i], key)) return obj->items[i];
    return NULL;
}

/* ------------------------------------------------------------------ socket I/O */
static int write_all(int fd, const char *p, size_t n) {
    while (n > 0) {
        const ssize_t w = write(fd, p, n);
        if (w < 0) { if (errno == EINTR) continue; return -1; }
        p += (size_t)w; n -= (size_t)w;
    }
    return 0;
}
static int write_str(int fd, const char *s) { return write_all(fd, s, strlen(s)); }

/* Reads one HTTP request: returns 0 on success with method/path (inside req) and body (malloc'd). */
static int read_request(int fd, sbuf *req, char **method, char **path, char **body, size_t *body_len) {
    const size_t max_head = 1u << 20, max_body = 64u << 20;
    size_t head_end = 0;
    for (;;) {
        sb_reserve(req, 8192);
        const ssize_t r = read(fd, req->p + req->len, req->cap - req->len - 1);
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        if (r == 0) return -1;
        req->len += (size_t)r; req->p[req->len] = 0;
        const char *e = strstr(req->p, "\r\n\r\n");
        if (e) { head_end = (size_t)(e - req->p) + 4; break; }
        if (req->len > max_head) return -1;
    }
    size_t content_length = 0;
    for (const char *h = req->p; h < req->p + head_end; ) {
        const char *eol = strstr(h, "\r\n");
        if (!eol) break;
        if (!strncasecmp(h, "Content-Length:", 15)) content_length = (size_t)strtoull(h + 15, NULL, 10);
        h = eol + 2;
    }
    if (content_length > max_body) return -1;
    while (req->len - head_end < content_length) {
        sb_reserve(req, content_length - (req->len - head_end) + 1);
        const ssize_t r = read(fd, req->p + req->len, req->cap - req->len - 1);
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        if (r == 0) return -1;
        req->len += (size_t)r; req->p[req->len] = 0;
    }
    char *line_end = strstr(req->p, "\r\n");
    *line_end = 0;
    *method = req->p;
    char *sp = strchr(req->p, ' ');
    if (!sp) return -1;
    *sp = 0;
    *path = sp + 1;
    char *sp2 = strchr(*path, ' ');
    if (sp2) *sp2 = 0;
    *body = malloc(content_length + 1);
    if (!*body) return -1;
    memcpy(*body, req->p + head_end, content_length);
    (*body)[content_length] = 0;
    *body_len = content_length;
    return 0;
}

static void send_json(int fd, int status, const char *status_text, const char *json) {
    char head[256];
    snprintf(head, sizeof(head), "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
             status, status_text, strlen(json));
    if (write_str(fd, head) == 0) write_str(fd, json);
}
static void send_error(int fd, int status, const char *status_text, const char *message) {
    sbuf b = {0};
    sb_puts(&b, "{\"error\":{\"message\":");
    sb_put_json_string(&b, message, strlen(message));
    sb_puts(&b, ",\"type\":\"invalid_request_error\"}}");
    send_json(fd, status, status_text, b.p);
    sb_free(&b);
}

/* ------------------------------------------------------------------ JSON output helpers */
static void sb_put_number(sbuf *b, double d) {
    char num[48];
    if (d == (double)(long long)d && d > -9.007199254740992e15 && d < 9.007199254740992e15) snprintf(num, sizeof(num), "%lld", (long long)d);
    else snprintf(num, sizeof(num), "%.17g", d);
    sb_puts(b, num);
}
/* Serialises a value. sorted: object keys in byte order (deltafin's deep_sort). compact: no spaces; otherwise ", " and
   ": " (deltafin's non-compact rendering of argument values). */
static void sb_put_jv(sbuf *b, const jv *v, bool sorted, bool compact) {
    if (!v) { sb_puts(b, "null"); return; }
    switch (v->type) {
    case JV_NULL: sb_puts(b, "null"); break;
    case JV_BOOL: sb_puts(b, v->boolean ? "true" : "false"); break;
    case JV_NUM: sb_put_number(b, v->num); break;
    case JV_STR: sb_put_json_string(b, v->str, strlen(v->str)); break;
    case JV_ARR:
        sb_putc(b, '[');
        for (int i = 0; i < v->n; i++) { if (i) sb_puts(b, compact ? "," : ", "); sb_put_jv(b, v->items[i], sorted, compact); }
        sb_putc(b, ']');
        break;
    case JV_OBJ: {
        int *order = malloc(sizeof(int) * (size_t)(v->n > 0 ? v->n : 1));
        if (!order) exit(1);
        for (int i = 0; i < v->n; i++) order[i] = i;
        if (sorted) {
            for (int i = 1; i < v->n; i++) {
                const int k = order[i]; int j = i;
                while (j > 0 && strcmp(v->keys[order[j - 1]], v->keys[k]) > 0) { order[j] = order[j - 1]; j--; }
                order[j] = k;
            }
        }
        sb_putc(b, '{');
        for (int i = 0; i < v->n; i++) {
            const int k = order[i];
            if (i) sb_puts(b, compact ? "," : ", ");
            sb_put_json_string(b, v->keys[k], strlen(v->keys[k]));
            sb_puts(b, compact ? ":" : ": ");
            sb_put_jv(b, v->items[k], sorted, compact);
        }
        sb_putc(b, '}');
        free(order);
        break;
    }
    }
}
static const char *xtml_type(const jv *v) {
    switch (v ? v->type : JV_NULL) {
    case JV_BOOL: return "boolean";
    case JV_NUM: return "number";
    case JV_STR: return "string";
    case JV_OBJ: return "object";
    case JV_ARR: return "array";
    default: return "null";
    }
}
static bool jv_truthy(const jv *v) {
    if (!v) return false;
    switch (v->type) {
    case JV_NULL: return false;
    case JV_BOOL: return v->boolean != 0;
    case JV_NUM: return v->num != 0.0;
    case JV_STR: return v->str && v->str[0];
    default: return v->n > 0;
    }
}

/* ------------------------------------------------------------------ K3 prompt builder (deltafin chat.rs rules)
   Control tokens are pushed by id; every other fragment is tokenized on its own, as deltafin encodes its segments:
   the tag, then " key", "=\"", the escaped value and "\"" per attribute. */
typedef struct { ds4_engine *e; ds4_tokens *out; int id_open, id_close, id_sep, id_eom; } k3p;
static void k3p_init(k3p *p, ds4_engine *e, ds4_tokens *out) {
    p->e = e; p->out = out;
    p->id_open = ds4_k3_special_id(e, "<|open|>"); p->id_close = ds4_k3_special_id(e, "<|close|>");
    p->id_sep = ds4_k3_special_id(e, "<|sep|>"); p->id_eom = ds4_k3_special_id(e, "<|end_of_msg|>");
}
static void k3p_text(k3p *p, const char *s) { if (s && s[0]) ds4_tokenize_text(p->e, s, p->out); }
static void k3p_control(k3p *p, int id, const char *spelling) { if (id >= 0) ds4_tokens_push(p->out, id); else k3p_text(p, spelling); }
static void k3p_sep(k3p *p) { k3p_control(p, p->id_sep, "<|sep|>"); }
static void k3p_open(k3p *p, const char *tag) { k3p_control(p, p->id_open, "<|open|>"); k3p_text(p, tag); }
static void k3p_attr(k3p *p, const char *key, const char *value) {
    char k[64];
    snprintf(k, sizeof(k), " %s", key);
    k3p_text(p, k);
    k3p_text(p, "=\"");
    sbuf esc = {0};
    sb_reserve(&esc, 16);
    for (const char *c = value ? value : ""; *c; c++) {
        if (*c == '&') sb_puts(&esc, "&amp;");
        else if (*c == '"') sb_puts(&esc, "&quot;");
        else sb_putc(&esc, *c);
    }
    k3p_text(p, esc.p);
    sb_free(&esc);
    k3p_text(p, "\"");
}
static void k3p_close(k3p *p, const char *tag) { k3p_control(p, p->id_close, "<|close|>"); k3p_text(p, tag); k3p_sep(p); }
static void k3p_eom(k3p *p) { k3p_control(p, p->id_eom, "<|end_of_msg|>"); }
static void k3p_name_attr(k3p *p, const jv *m) {
    const jv *name = jv_get(m, "name");
    if (name && name->type == JV_STR && name->str[0]) k3p_attr(p, "name", name->str);
    else if (name && jv_truthy(name)) { sbuf t = {0}; sb_put_jv(&t, name, false, false); k3p_attr(p, "name", t.p); sb_free(&t); }
}
/* <|open|>message role="system" type="TYPE"<|sep|>BODY<|close|>message<|sep|><|end_of_msg|> */
static void k3p_system(k3p *p, const char *type, const char *body) {
    k3p_open(p, "message"); k3p_attr(p, "role", "system"); k3p_attr(p, "type", type); k3p_sep(p);
    k3p_text(p, body);
    k3p_close(p, "message"); k3p_eom(p);
}
static void k3p_tool_declare(k3p *p, const jv *tools, bool dynamic) {
    sbuf body = {0};
    sb_puts(&body, dynamic
            ? "## New Tools Available\nThe system dynamically extends the toolset via lazy-loading.\nYou have access to all existing and extended tools.\nHere are the specs for the extended tools.\n\n```json\n"
            : "# Tools\nHere are the available tools, described in JSONSchema.\n\n```json\n");
    sb_put_jv(&body, tools, true, true);
    sb_puts(&body, "\n```");
    k3p_system(p, "tool-declare", body.p);
    sb_free(&body);
}
static char *content_text(const jv *content) {
    /* a string, or OpenAI content parts [{"type":"text","text":...}]; other values are rendered as JSON */
    sbuf b = {0};
    sb_reserve(&b, 16);
    if (!content || content->type == JV_NULL) return b.p;
    if (content->type == JV_STR) { sb_puts(&b, content->str); return b.p; }
    if (content->type == JV_ARR) {
        for (int i = 0; i < content->n; i++) {
            const jv *part = content->items[i];
            const jv *t = jv_get(part, "text");
            if (t && t->type == JV_STR) sb_puts(&b, t->str);
        }
        return b.p;
    }
    sb_put_jv(&b, content, false, false);
    return b.p;
}
static const jv *call_function(const jv *call) {
    const jv *fn = jv_get(call, "function");
    return fn && fn->type == JV_OBJ ? fn : call;
}
/* assistant history: <|open|>tools<|sep|> (<|open|>call tool="NAME" index="i"<|sep|> arguments <|close|>call<|sep|>)* <|close|>tools<|sep|> */
static void k3p_tool_calls(k3p *p, const jv *calls) {
    k3p_open(p, "tools"); k3p_sep(p);
    for (int i = 0; i < calls->n; i++) {
        const jv *fn = call_function(calls->items[i]);
        const jv *name = jv_get(fn, "name");
        if (!name || name->type != JV_STR) continue;
        char idx[16];
        snprintf(idx, sizeof(idx), "%d", i + 1);
        k3p_open(p, "call"); k3p_attr(p, "tool", name->str); k3p_attr(p, "index", idx); k3p_sep(p);
        const jv *args = jv_get(fn, "arguments");
        jv *parsed = NULL;
        const jv *obj = NULL;
        const char *raw = NULL;
        if (args && args->type == JV_OBJ) obj = args;
        else if (args && args->type == JV_STR && *jp_ws(args->str)) {
            const char *q = args->str;
            parsed = jp_value(&q, 0);
            if (parsed && parsed->type == JV_OBJ && !*jp_ws(q)) obj = parsed;
            else raw = args->str;
        }
        if (obj) {
            for (int k = 0; k < obj->n; k++) {
                const jv *val = obj->items[k];
                k3p_open(p, "argument"); k3p_attr(p, "key", obj->keys[k]); k3p_attr(p, "type", xtml_type(val)); k3p_sep(p);
                if (val->type == JV_STR) k3p_text(p, val->str);
                else { sbuf t = {0}; sb_put_jv(&t, val, false, false); k3p_text(p, t.p); sb_free(&t); }
                k3p_close(p, "argument");
            }
        } else if (raw) {
            k3p_open(p, "json"); k3p_attr(p, "type", "object"); k3p_sep(p);
            k3p_text(p, raw);
            k3p_close(p, "json");
        }
        jv_free(parsed);
        k3p_close(p, "call");
    }
    k3p_close(p, "tools");
}
static void k3p_assistant(k3p *p, const jv *m, bool thinking) {
    k3p_open(p, "message"); k3p_attr(p, "role", "assistant"); k3p_name_attr(p, m); k3p_sep(p);
    if (thinking) {
        const jv *reasoning = jv_get(m, "reasoning_content");
        if (!jv_truthy(reasoning)) reasoning = jv_get(m, "reasoning");
        k3p_open(p, "think"); k3p_sep(p);
        if (reasoning && reasoning->type == JV_STR && *jp_ws(reasoning->str)) k3p_text(p, reasoning->str);
        k3p_close(p, "think");
    }
    k3p_open(p, "response"); k3p_sep(p);
    char *text = content_text(jv_get(m, "content"));
    k3p_text(p, text);
    free(text);
    k3p_close(p, "response");
    const jv *calls = jv_get(m, "tool_calls");
    if (calls && calls->type == JV_ARR && calls->n > 0) k3p_tool_calls(p, calls);
    k3p_close(p, "message"); k3p_eom(p);
}
/* tool result: <|open|>message role="tool" tool="NAME" index="i"<|sep|>CONTENT<|close|>message<|sep|><|end_of_msg|>
   NAME/index come from tool_call_id matched against the preceding assistant's calls, else from name/tool and the
   position within the run of tool messages (deltafin's rule). */
static void k3p_tool_result(k3p *p, const jv *m, const jv *prev_calls, int run_index) {
    const char *name = NULL;
    int index = run_index;
    const jv *id = jv_get(m, "tool_call_id");
    if (!id) id = jv_get(m, "id");
    if (id && id->type == JV_STR && prev_calls) {
        for (int i = 0; i < prev_calls->n; i++) {
            const jv *cid = jv_get(prev_calls->items[i], "id");
            if (cid && cid->type == JV_STR && !strcmp(cid->str, id->str)) {
                const jv *n = jv_get(call_function(prev_calls->items[i]), "name");
                if (n && n->type == JV_STR) { name = n->str; index = i + 1; }
                break;
            }
        }
    }
    if (!name) {
        const jv *n = jv_get(m, "tool");
        if (!n || n->type != JV_STR || !n->str[0]) n = jv_get(m, "name");
        if (n && n->type == JV_STR && n->str[0]) name = n->str;
    }
    if (!name && prev_calls && run_index - 1 < prev_calls->n) {
        const jv *n = jv_get(call_function(prev_calls->items[run_index - 1]), "name");
        if (n && n->type == JV_STR) name = n->str;
    }
    char idx[16];
    snprintf(idx, sizeof(idx), "%d", index);
    k3p_open(p, "message"); k3p_attr(p, "role", "tool"); k3p_attr(p, "tool", name ? name : "unknown"); k3p_attr(p, "index", idx); k3p_sep(p);
    char *text = content_text(jv_get(m, "content"));
    k3p_text(p, text);
    free(text);
    k3p_close(p, "message"); k3p_eom(p);
}
static void k3p_effort(k3p *p, const char *effort) {
    char text[512];
    snprintf(text, sizeof(text),
             "`thinking_effort` guides on how much to think in your thinking channel (not including the response "
             "channel), supported values include `low`, `medium`, `high`, and `max`.\nNow the system is invoked with "
             "`thinking_effort=%s`.", effort);
    k3p_system(p, "thinking-effort", text);
}
/* Renders the whole request into tokens. Order (deltafin): tool declaration, thinking-effort, the messages,
   tool-choice, response-format, the assistant prefix. */
static void k3_build_prompt(ds4_engine *e, const jv *root, const jv *messages, bool thinking, const char *effort,
                            bool *tools_declared, ds4_tokens *out) {
    k3p p;
    k3p_init(&p, e, out);
    const jv *tools = jv_get(root, "tools");
    *tools_declared = tools && tools->type == JV_ARR && tools->n > 0;
    if (*tools_declared) k3p_tool_declare(&p, tools, false);
    if (thinking) k3p_effort(&p, effort);
    const jv *prev_calls = NULL;
    int tool_run = 0;
    int prefix_boundary = 0;        /* step 39b: tokens up to the end of the leading system message(s) */
    bool leading_system = true;
    for (int i = 0; i < messages->n; i++) {
        const jv *m = messages->items[i];
        if (!m || m->type != JV_OBJ) continue;
        const jv *role = jv_get(m, "role");
        const char *r = role && role->type == JV_STR ? role->str : "user";
        if (leading_system && strcmp(r, "system")) leading_system = false;
        if (!strcmp(r, "assistant")) {
            const jv *calls = jv_get(m, "tool_calls");
            prev_calls = calls && calls->type == JV_ARR ? calls : NULL;
            tool_run = 0;
            k3p_assistant(&p, m, thinking);
        } else if (!strcmp(r, "tool")) {
            tool_run++;
            k3p_tool_result(&p, m, prev_calls, tool_run);
        } else if (!strcmp(r, "system") && jv_truthy(jv_get(m, "tools"))) {
            k3p_tool_declare(&p, jv_get(m, "tools"), true);
            *tools_declared = true;
        } else {
            const char *rr = !strcmp(r, "system") ? "system" : "user";
            k3p_open(&p, "message"); k3p_attr(&p, "role", rr); k3p_name_attr(&p, m); k3p_sep(&p);
            char *text = content_text(jv_get(m, "content"));
            k3p_text(&p, text);
            free(text);
            k3p_close(&p, "message"); k3p_eom(&p);
        }
        if (leading_system) prefix_boundary = out->len;
    }
    ds4_k3_note_prefix_boundary(prefix_boundary);
    const jv *choice = jv_get(root, "tool_choice");
    if (choice && choice->type == JV_STR && !strcmp(choice->str, "none"))
        k3p_system(&p, "tool-choice", "The system is invoked with `tool_choice=none`.\nYou MUST NOT call any tools in the next message.");
    else if (choice && ((choice->type == JV_STR && !strcmp(choice->str, "required")) || choice->type == JV_OBJ))
        k3p_system(&p, "tool-choice", "The system is invoked with `tool_choice=required`.\nYou MUST call tools in the next message.");
    const jv *fmt = jv_get(root, "response_format");
    const jv *ft = fmt && fmt->type == JV_OBJ ? jv_get(fmt, "type") : fmt;
    if (ft && ft->type == JV_STR && !strcmp(ft->str, "json_object")) {
        k3p_system(&p, "response-format", "The system is invoked with `response_format=json_object`.\nYour response must be raw JSON data without markdown code blocks (```json) or any additional formatting.");
    } else if (ft && ft->type == JV_STR && !strcmp(ft->str, "json_schema")) {
        const jv *js = jv_get(fmt, "json_schema");
        const jv *schema = js && js->type == JV_OBJ ? jv_get(js, "schema") : NULL;
        if (!schema) schema = jv_get(fmt, "schema");
        sbuf body = {0};
        sb_puts(&body, "The system is invoked with `response_format=json_schema`.\nYour response must be raw JSON data without markdown code blocks (```json) or any additional formatting.\nThe JSON data must match the following schema:\n```json\n");
        sb_put_jv(&body, schema, true, true);
        sb_puts(&body, "\n```");
        k3p_system(&p, "response-format", body.p);
        sb_free(&body);
    }
    k3p_open(&p, "message"); k3p_attr(&p, "role", "assistant"); k3p_sep(&p);
    k3p_open(&p, thinking ? "think" : "response"); k3p_sep(&p);
}

/* ------------------------------------------------------------------ generation plumbing */
typedef struct {
    ds4_engine *e;
    int fd;
    bool stream;
    bool failed;
    int produced;
    sbuf text;        /* the whole answer */
    sbuf reasoning;   /* the think block (thinking requests) */
    sbuf trailer;     /* tools mode: everything generated after the response block closed, raw (tags included) */
    sbuf pending_reasoning;
    sbuf pending;     /* streamed bytes not yet sent (incomplete UTF-8 tail) */
    char id[48];
    long created;
} k3_request;

static size_t utf8_complete_prefix(const char *s, size_t n) {
    /* length of the longest prefix that does not end inside a multi-byte sequence */
    size_t i = n;
    while (i > 0) {
        const unsigned char c = (unsigned char)s[i - 1];
        if ((c & 0xC0) != 0x80) {
            size_t need = c < 0x80 ? 1 : c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
            return (n - (i - 1)) >= need ? n : i - 1;
        }
        if (n - i >= 3) return n;   /* malformed: let it through */
        i--;
    }
    return n;
}
/* one SSE chunk; delta_body is the JSON inside delta:{...} (may be empty) */
static void stream_chunk_delta(k3_request *r, const char *delta_body, const char *finish_reason) {
    sbuf b = {0};
    sb_puts(&b, "data: {\"id\":\""); sb_puts(&b, r->id);
    sb_puts(&b, "\",\"object\":\"chat.completion.chunk\",\"created\":");
    char num[32]; snprintf(num, sizeof(num), "%ld", r->created); sb_puts(&b, num);
    sb_puts(&b, ",\"model\":\"kimi-k3\",\"choices\":[{\"index\":0,\"delta\":{");
    if (delta_body) sb_puts(&b, delta_body);
    sb_puts(&b, "},\"finish_reason\":");
    if (finish_reason) { sb_putc(&b, '"'); sb_puts(&b, finish_reason); sb_putc(&b, '"'); } else sb_puts(&b, "null");
    sb_puts(&b, "}]}\n\n");
    if (write_all(r->fd, b.p, b.len) != 0) { r->failed = true; ds4_k3_cancel(); }
    sb_free(&b);
}
static void stream_chunk_kind(k3_request *r, const char *field, const char *content, size_t n, const char *finish_reason) {
    sbuf d = {0};
    if (content) { sb_putc(&d, '"'); sb_puts(&d, field); sb_puts(&d, "\":"); sb_put_json_string(&d, content, n); }
    stream_chunk_delta(r, d.p ? d.p : "", finish_reason);
    sb_free(&d);
}
static void flush_pending(k3_request *r, sbuf *pending, const char *field, bool all) {
    const size_t ok = all ? pending->len : utf8_complete_prefix(pending->p, pending->len);
    if (ok == 0) return;
    stream_chunk_kind(r, field, pending->p, ok, NULL);
    memmove(pending->p, pending->p + ok, pending->len - ok);
    pending->len -= ok;
    pending->p[pending->len] = 0;
}
static void emit_token(void *ud, int token) {
    k3_request *r = ud;
    size_t n = 0;
    char *piece = ds4_token_text(r->e, token, &n);
    r->produced++;
    if (ds4_k3_response_closed()) {   /* tools mode trailer: collected raw, parsed after generation */
        if (piece && n) sb_putn(&r->trailer, piece, n);
        free(piece);
        return;
    }
    const bool think = ds4_k3_current_block() == 1;
    if (piece && n) {
        sb_putn(think ? &r->reasoning : &r->text, piece, n);
        if (r->stream && !r->failed) {
            if (think) { sb_putn(&r->pending_reasoning, piece, n); flush_pending(r, &r->pending_reasoning, "reasoning_content", false); }
            else { sb_putn(&r->pending, piece, n); flush_pending(r, &r->pending, "content", false); }
        }
    }
    free(piece);
}
static void generation_done(void *ud) { (void)ud; }

/* ------------------------------------------------------------------ tool calls emitted by the model
   <|open|>tools<|sep|><|open|>call tool="NAME" index="1"<|sep|>
     <|open|>argument key="K" type="string"<|sep|>VALUE<|close|>argument<|sep|> ...   or
     <|open|>json type="object"<|sep|>{...}<|close|>json<|sep|>
   <|close|>call<|sep|>...<|close|>tools<|sep|> */
typedef struct { char *name; char *args; } k3_call;
static const char *find_str(const char *s, const char *end, const char *needle) {
    const size_t n = strlen(needle);
    for (const char *p = s; p + n <= end; p++) if (!memcmp(p, needle, n)) return p;
    return NULL;
}
static char *tag_attr(const char *head, const char *head_end, const char *key) {
    const size_t kn = strlen(key);
    for (const char *p = head; p + kn + 2 <= head_end; p++) {
        if ((p == head || p[-1] == ' ') && !memcmp(p, key, kn) && p[kn] == '=' && p[kn + 1] == '"') {
            const char *v = p + kn + 2, *q = v;
            while (q < head_end && *q != '"') q++;
            sbuf b = {0};
            sb_reserve(&b, 16);
            for (const char *c = v; c < q; c++) {
                if (q - c >= 6 && !memcmp(c, "&quot;", 6)) { sb_putc(&b, '"'); c += 5; }
                else if (q - c >= 5 && !memcmp(c, "&amp;", 5)) { sb_putc(&b, '&'); c += 4; }
                else sb_putc(&b, *c);
            }
            return b.p;
        }
    }
    return NULL;
}
static void trim_span(const char **s, const char **e) {
    while (*s < *e && ((unsigned char)**s == ' ' || **s == '\n' || **s == '\r' || **s == '\t')) (*s)++;
    while (*e > *s && ((unsigned char)(*e)[-1] == ' ' || (*e)[-1] == '\n' || (*e)[-1] == '\r' || (*e)[-1] == '\t')) (*e)--;
}
static bool span_is_json(const char *s, const char *e) {
    char *tmp = strndup(s, (size_t)(e - s));
    if (!tmp) return false;
    const char *p = tmp;
    jv *v = jp_value(&p, 0);
    const bool ok = v != NULL && !*jp_ws(p);
    jv_free(v);
    free(tmp);
    return ok;
}
static int parse_tool_calls(const char *t, size_t n, k3_call **out) {
    const char *end = t + n;
    const char *p = find_str(t, end, "<|open|>tools<|sep|>");
    *out = NULL;
    if (!p) return 0;
    p += strlen("<|open|>tools<|sep|>");
    const char *tools_end = find_str(p, end, "<|close|>tools<|sep|>");
    if (!tools_end) tools_end = end;
    k3_call *calls = NULL;
    int n_calls = 0;
    for (;;) {
        const char *c = find_str(p, tools_end, "<|open|>call");
        if (!c) break;
        const char *head = c + strlen("<|open|>");
        const char *head_end = find_str(head, tools_end, "<|sep|>");
        if (!head_end) break;
        char *name = tag_attr(head, head_end, "tool");
        const char *body = head_end + strlen("<|sep|>");
        const char *body_end = find_str(body, tools_end, "<|close|>call<|sep|>");
        if (!body_end) body_end = tools_end;
        sbuf args = {0};
        sb_reserve(&args, 32);
        int n_args = 0;
        bool raw_json = false;
        const char *q = body;
        for (;;) {
            const char *o = find_str(q, body_end, "<|open|>");
            if (!o) break;
            const char *h = o + strlen("<|open|>");
            const char *he = find_str(h, body_end, "<|sep|>");
            if (!he) break;
            const char *v = he + strlen("<|sep|>");
            if (he - h >= 8 && !memcmp(h, "argument", 8)) {
                const char *ve = find_str(v, body_end, "<|close|>argument<|sep|>");
                const char *next = ve ? ve + strlen("<|close|>argument<|sep|>") : body_end;
                if (!ve) ve = body_end;
                char *key = tag_attr(h, he, "key");
                char *type = tag_attr(h, he, "type");
                if (!raw_json) {
                    sb_putc(&args, n_args ? ',' : '{');
                    sb_put_json_string(&args, key ? key : "", key ? strlen(key) : 0);
                    sb_putc(&args, ':');
                    bool as_string = !type || !strcmp(type, "string");
                    if (!as_string) {
                        const char *vs = v, *vend = ve;
                        trim_span(&vs, &vend);
                        if (vs < vend && span_is_json(vs, vend)) sb_putn(&args, vs, (size_t)(vend - vs));
                        else as_string = true;
                    }
                    if (as_string) sb_put_json_string(&args, v, (size_t)(ve - v));
                    n_args++;
                }
                free(key); free(type);
                q = next;
            } else if (he - h >= 4 && !memcmp(h, "json", 4)) {
                const char *ve = find_str(v, body_end, "<|close|>json<|sep|>");
                const char *next = ve ? ve + strlen("<|close|>json<|sep|>") : body_end;
                if (!ve) ve = body_end;
                const char *vs = v, *vend = ve;
                trim_span(&vs, &vend);
                args.len = 0; args.p[0] = 0;
                sb_putn(&args, vs, (size_t)(vend - vs));
                raw_json = true;
                q = next;
            } else {
                q = v;
            }
        }
        if (!raw_json) { if (n_args) sb_putc(&args, '}'); else sb_puts(&args, "{}"); }
        calls = realloc(calls, sizeof(k3_call) * (size_t)(n_calls + 1));
        if (!calls) exit(1);
        calls[n_calls].name = name ? name : strdup("");
        calls[n_calls].args = args.p;
        n_calls++;
        p = body_end == tools_end ? tools_end : body_end + strlen("<|close|>call<|sep|>");
    }
    *out = calls;
    return n_calls;
}
static void put_tool_call_json(sbuf *b, const k3_request *r, const k3_call *c, int i, bool with_index) {
    char num[32];
    sb_putc(b, '{');
    if (with_index) { snprintf(num, sizeof(num), "\"index\":%d,", i); sb_puts(b, num); }
    snprintf(num, sizeof(num), "\"id\":\"call_%d_", i + 1); sb_puts(b, num); sb_puts(b, r->id + 12);
    sb_puts(b, "\",\"type\":\"function\",\"function\":{\"name\":");
    sb_put_json_string(b, c->name, strlen(c->name));
    sb_puts(b, ",\"arguments\":");
    sb_put_json_string(b, c->args, strlen(c->args));
    sb_puts(b, "}}");
}

static void handle_chat(ds4_engine *e, int fd, const char *body, int ctx_size, int default_max_tokens) {
    const char *p = body;
    jv *root = jp_value(&p, 0);
    if (!root || root->type != JV_OBJ) { jv_free(root); send_error(fd, 400, "Bad Request", "request body is not a JSON object"); return; }
    const jv *messages = jv_get(root, "messages");
    if (!messages || messages->type != JV_ARR || messages->n == 0) { jv_free(root); send_error(fd, 400, "Bad Request", "messages must be a non-empty array"); return; }
    int max_tokens = default_max_tokens;
    const jv *mt = jv_get(root, "max_tokens");
    if (!mt) mt = jv_get(root, "max_completion_tokens");
    if (mt && mt->type == JV_NUM && mt->num > 0) max_tokens = (int)mt->num;
    float temperature = 0.0f, top_p = 1.0f;
    int top_k = 0;
    uint64_t seed = 0;
    {
        const jv *v;
        if ((v = jv_get(root, "temperature")) && v->type == JV_NUM) temperature = (float)v->num;
        if ((v = jv_get(root, "top_p")) && v->type == JV_NUM) top_p = (float)v->num;
        if ((v = jv_get(root, "top_k")) && v->type == JV_NUM) top_k = (int)v->num;
        if ((v = jv_get(root, "seed")) && v->type == JV_NUM) seed = (uint64_t)v->num;
        if (!seed) seed = (uint64_t)time(NULL) * 1000003ull ^ (uint64_t)getpid();
        ds4_k3_set_sampling(temperature, top_p, top_k, seed);
    }
    const jv *st = jv_get(root, "stream");
    const bool stream = st && st->type == JV_BOOL && st->boolean;
    /* thinking: "thinking": true, or "reasoning_effort" set to anything but "none"; default from DS4_K3_THINK.
       The effort rendered into the prompt is max unless reasoning_effort names low/high (medium counts as high). */
    int thinking = getenv("DS4_K3_THINK") && getenv("DS4_K3_THINK")[0] != '0';
    const char *effort = "max";
    const jv *th = jv_get(root, "thinking");
    if (th && th->type == JV_BOOL) thinking = th->boolean;
    const jv *re = jv_get(root, "reasoning_effort");
    if (re && re->type == JV_STR) {
        thinking = strcmp(re->str, "none") != 0;
        if (!strcmp(re->str, "low")) effort = "low";
        else if (!strcmp(re->str, "medium") || !strcmp(re->str, "high")) effort = "high";
    }

    ds4_tokens prompt = {0};
    bool tools_declared = false;
    k3_build_prompt(e, root, messages, thinking != 0, effort, &tools_declared, &prompt);
    jv_free(root);
    if (prompt.len == 0 || prompt.len >= ctx_size - 1) {
        char msg[128];
        snprintf(msg, sizeof(msg), "prompt is %d tokens; this K3 build holds %d positions", prompt.len, ctx_size);
        ds4_tokens_free(&prompt);
        send_error(fd, 400, "Bad Request", msg);
        return;
    }
    if (max_tokens > ctx_size - 1 - prompt.len) max_tokens = ctx_size - 1 - prompt.len;

    k3_request r;
    memset(&r, 0, sizeof(r));
    r.e = e; r.fd = fd; r.stream = stream; r.created = (long)time(NULL);
    snprintf(r.id, sizeof(r.id), "chatcmpl-k3-%ld-%d", r.created, (int)(getpid() & 0xffff));
    sb_reserve(&r.text, 256); sb_reserve(&r.pending, 64); sb_reserve(&r.reasoning, 64); sb_reserve(&r.pending_reasoning, 64);
    sb_reserve(&r.trailer, 64);
    if (stream) {
        if (write_str(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n") != 0) {
            ds4_tokens_free(&prompt); sb_free(&r.text); sb_free(&r.pending); sb_free(&r.reasoning); sb_free(&r.pending_reasoning); sb_free(&r.trailer); return;
        }
        stream_chunk_delta(&r, "\"role\":\"assistant\",\"content\":\"\"", NULL);
    }
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    ds4_k3_set_tools_mode(tools_declared);
    const int rc = ds4_engine_generate_argmax(e, &prompt, max_tokens, ctx_size, emit_token, generation_done, &r, NULL, NULL);
    ds4_k3_set_tools_mode(0);
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    const double secs = (double)(ts1.tv_sec - ts0.tv_sec) + (double)(ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    k3_call *calls = NULL;
    const int n_calls = r.trailer.len ? parse_tool_calls(r.trailer.p, r.trailer.len, &calls) : 0;
    const char *finish = n_calls > 0 ? "tool_calls" : r.produced >= max_tokens ? "length" : "stop";
    fprintf(stderr, "ds4: K3 serve: %d prompt tokens, %d generated in %.1f s (%.2f tok/s), %s%s%s%s%s, %s\n", prompt.len,
            r.produced, secs, secs > 0 ? r.produced / secs : 0.0, finish, thinking ? ", thinking" : "",
            tools_declared ? ", tools" : "", rc != 0 ? ", generation error" : "", r.failed ? ", client gone" : "",
            temperature > 0.0f ? "sampled" : "greedy");
    if (temperature > 0.0f) {
        fprintf(stderr, "ds4: K3 serve: sampling temperature %.2f top_p %.2f top_k %d seed %llu\n", temperature, top_p, top_k,
                (unsigned long long)seed);
    }
    if (tools_declared && getenv("DS4_K3_TRACE_TOOLS")) fprintf(stderr, "ds4: K3 serve: trailer: %s\n", r.trailer.p ? r.trailer.p : "");
    for (int i = 0; i < n_calls; i++) fprintf(stderr, "ds4: K3 serve: tool call %d: %s %s\n", i + 1, calls[i].name, calls[i].args);
    if (stream) {
        if (!r.failed) {
            if (r.pending_reasoning.len) flush_pending(&r, &r.pending_reasoning, "reasoning_content", true);
            if (r.pending.len) flush_pending(&r, &r.pending, "content", true);
            for (int i = 0; i < n_calls; i++) {
                sbuf d = {0};
                sb_puts(&d, "\"tool_calls\":[");
                put_tool_call_json(&d, &r, &calls[i], i, true);
                sb_putc(&d, ']');
                stream_chunk_delta(&r, d.p, NULL);
                sb_free(&d);
            }
            stream_chunk_delta(&r, "", finish);
            write_str(fd, "data: [DONE]\n\n");
        }
    } else {
        sbuf b = {0};
        char num[64];
        sb_puts(&b, "{\"id\":\""); sb_puts(&b, r.id);
        sb_puts(&b, "\",\"object\":\"chat.completion\",\"created\":"); snprintf(num, sizeof(num), "%ld", r.created); sb_puts(&b, num);
        sb_puts(&b, ",\"model\":\"kimi-k3\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
        if (r.text.len == 0 && n_calls > 0) sb_puts(&b, "null");
        else sb_put_json_string(&b, r.text.p ? r.text.p : "", r.text.len);
        if (r.reasoning.len) { sb_puts(&b, ",\"reasoning_content\":"); sb_put_json_string(&b, r.reasoning.p, r.reasoning.len); }
        if (n_calls > 0) {
            sb_puts(&b, ",\"tool_calls\":[");
            for (int i = 0; i < n_calls; i++) { if (i) sb_putc(&b, ','); put_tool_call_json(&b, &r, &calls[i], i, false); }
            sb_putc(&b, ']');
        }
        sb_puts(&b, "},\"finish_reason\":\""); sb_puts(&b, finish); sb_puts(&b, "\"}],\"usage\":{\"prompt_tokens\":");
        snprintf(num, sizeof(num), "%d,\"completion_tokens\":%d,\"total_tokens\":%d}}", prompt.len, r.produced, prompt.len + r.produced);
        sb_puts(&b, num);
        send_json(fd, 200, "OK", b.p);
        sb_free(&b);
    }
    for (int i = 0; i < n_calls; i++) { free(calls[i].name); free(calls[i].args); }
    free(calls);
    ds4_tokens_free(&prompt);
    sb_free(&r.text); sb_free(&r.pending); sb_free(&r.reasoning); sb_free(&r.pending_reasoning); sb_free(&r.trailer);
}

int ds4_k3_serve(ds4_engine *e, const char *host, int port, int ctx_size, int default_max_tokens) {
    if (!e || !host || port <= 0) return 1;
    if (ctx_size <= 0) ctx_size = 2048;
    if (default_max_tokens <= 0) default_max_tokens = 512;
    signal(SIGPIPE, SIG_IGN);
    const int lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) { perror("ds4: K3 serve: socket"); return 1; }
    const int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) { fprintf(stderr, "ds4: K3 serve: bad host %s\n", host); close(lfd); return 1; }
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) != 0 || listen(lfd, 16) != 0) { perror("ds4: K3 serve: bind/listen"); close(lfd); return 1; }
    /* Warm-up: the spine is staged from disk on first use (about 48 GB at the RAID's rate, 2-4 s), so take that
       hit now rather than on the first request. */
    {
        struct timespec w0, w1;
        clock_gettime(CLOCK_MONOTONIC, &w0);
        ds4_tokens warm = {0};
        ds4_tokenize_text(e, "Hello", &warm);
        if (warm.len > 0) {
            if (ds4_engine_generate_argmax(e, &warm, 1, ctx_size, NULL, NULL, NULL, NULL, NULL) != 0) {
                fprintf(stderr, "ds4: K3 serve: warm-up generation failed\n");
            }
        }
        ds4_tokens_free(&warm);
        clock_gettime(CLOCK_MONOTONIC, &w1);
        fprintf(stderr, "ds4: K3 serve: warm-up done in %.1f s\n",
                (double)(w1.tv_sec - w0.tv_sec) + (double)(w1.tv_nsec - w0.tv_nsec) * 1e-9);
    }
    fprintf(stderr, "ds4: K3 serve: listening on http://%s:%d/v1 (ctx %d, default max_tokens %d, tools/thinking/sampling, one request at a time)\n",
            host, port, ctx_size, default_max_tokens);
    for (;;) {
        struct sockaddr_in peer;
        socklen_t plen = sizeof(peer);
        const int fd = accept(lfd, (struct sockaddr *)&peer, &plen);
        if (fd < 0) { if (errno == EINTR) continue; perror("ds4: K3 serve: accept"); break; }
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        sbuf req = {0};
        char *method = NULL, *path = NULL, *body = NULL;
        size_t body_len = 0;
        if (read_request(fd, &req, &method, &path, &body, &body_len) == 0) {
            if (!strcmp(method, "GET") && (!strcmp(path, "/v1/models") || !strcmp(path, "/models"))) {
                send_json(fd, 200, "OK", "{\"object\":\"list\",\"data\":[{\"id\":\"kimi-k3\",\"object\":\"model\",\"owned_by\":\"local\"}]}");
            } else if (!strcmp(method, "GET") && (!strcmp(path, "/health") || !strcmp(path, "/"))) {
                send_json(fd, 200, "OK", "{\"status\":\"ok\",\"model\":\"kimi-k3\"}");
            } else if (!strcmp(method, "POST") && (!strcmp(path, "/v1/chat/completions") || !strcmp(path, "/chat/completions"))) {
                handle_chat(e, fd, body ? body : "", ctx_size, default_max_tokens);
            } else if (!strcmp(method, "OPTIONS")) {
                write_str(fd, "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n");
            } else {
                send_error(fd, 404, "Not Found", "unknown route");
            }
        }
        free(body);
        sb_free(&req);
        close(fd);
    }
    close(lfd);
    return 0;
}

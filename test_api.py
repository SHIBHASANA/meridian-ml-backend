import json, app
c = app.app.test_client()
N = dict(hr=66, spo2=98, temp=36.8, sys=116, rr=14, sleep=7, stress=4, hyd=82)
def post(body, raw=None):
    return c.post("/api/analyze", data=raw if raw is not None else json.dumps(body), content_type="application/json")
R = []
def ck(n, ok, d=""): R.append(ok); print(("PASS " if ok else "FAIL ") + n, d)
h = c.get("/api/health").get_json(); ck("health: model loaded", h["model_loaded"], h["version"])
r = post({"crew": "t", "features": N}).get_json(); ck("normal reading -> normal_pattern", r["classification"] == "normal_pattern", (r["score"], r["normality_percentile"]))
for name, f in {"cardiac-like": dict(N, hr=140, sys=152, rr=24, spo2=95), "fever+dehydration-like": dict(N, temp=38.8, hyd=44, hr=96),
                "hypoxia-like": dict(N, spo2=90, rr=27, hr=102), "sleep+stress-like": dict(N, sleep=4.5, stress=8.5, hr=78)}.items():
    r = post({"features": f}).get_json(); print("   ", name, r["label"], r["score"], [c_["feature"] for c_ in r["contributors"]])
    if name.startswith("sleep"):
        print("    KNOWN LIMITATION: sleep/stress-only shifts are often missed (see metrics.json per-kind recall); classified:", r["classification"])
    else:
        ck("unusual pattern flagged: " + name, r["classification"] != "normal_pattern")
r = post({"features": dict(N, hr=140, sys=152, rr=24, spo2=95)}).get_json(); ck("contributors include hr", "hr" in [x["feature"] for x in r["contributors"]])
ck("missing feature -> 422", post({"features": {k: v for k, v in N.items() if k != "hr"}}).status_code == 422)
ck("out-of-range -> 422", post({"features": dict(N, spo2=140)}).status_code == 422)
ck("string value -> 422", post({"features": dict(N, hr="fast")}).status_code == 422)
ck("bool value -> 422", post({"features": dict(N, hr=True)}).status_code == 422)
ck("NaN value -> 422", post(None, raw='{"features":{"hr":NaN,"spo2":98,"temp":36.8,"sys":116,"rr":14,"sleep":7,"stress":4,"hyd":80}}').status_code == 422)
ck("non-JSON -> 400", post(None, raw="hello").status_code == 400)
ck("no features object -> 400", post({"x": 1}).status_code == 400)
ck("CORS header present", c.get("/api/health").headers.get("Access-Control-Allow-Origin") == "*")
print("ALL PASS" if all(R) else "SOME FAIL")

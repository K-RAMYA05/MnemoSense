import os, json, datetime as dt
from utils import load_config, TRANS, RECS
from rag_store import delete_ids, _load_meta
def apply_forgetting():
    cfg=load_config(); pol=cfg.get("policies",{})
    keep_txt_days=int(pol.get("keep_days_text",90)); keep_wav_days=int(pol.get("keep_days_audio",14))
    protect=set(pol.get("protect_tags",[]))
    cutoff_txt=dt.datetime.now()-dt.timedelta(days=keep_txt_days)
    meta=_load_meta(); to_delete=[]
    for d in meta["docs"]:
        try: ts=dt.datetime.fromisoformat(d["ts"])
        except: continue
        if ts < cutoff_txt and not (set(d.get("tags",[])) & protect): to_delete.append(d["id"])
    if to_delete: delete_ids(to_delete)
    return {"deleted_doc_ids": to_delete}

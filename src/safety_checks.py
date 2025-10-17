import datetime as dt
from utils import TRANS, load_config, read_jsonl
from tts import speak
def _today_str(): return dt.datetime.now().date().isoformat()
def _events_today():
    evs = read_jsonl(TRANS/"events.jsonl"); today=_today_str()
    return [e for e in evs if e.get("ts","").startswith(today)]
def _did_keywords(evs, keywords):
    big = " ".join([e.get("text","") for e in evs]).lower()
    return any(k.lower() in big for k in keywords)
def run_checks(speak_out=True):
    """
    Returns a human message string regardless of speak_out.
    When speak_out=True, it also speaks the message.
    """
    cfg = load_config()
    user = cfg.get("user_name","Friend")
    evs = _events_today()
    missing = []

    for key, rule in cfg.get("schedules",{}).items():
        kw = rule.get("keywords",[])
        if not _did_keywords(evs, kw):
            missing.append(key)

    if not missing:
        msg = f"All good today, {user}. Nice job staying on track."
    else:
        human = ", ".join(missing)
        msg = f"Reminder, {user}: please complete — {human}."

    if speak_out:
        speak(msg)
    return msg


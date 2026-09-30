"""Bandit static review; this is not a runtime sandbox."""
import json,subprocess,sys,tempfile
from pathlib import Path
def security_score(code):
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/"solution.py"; path.write_text(code or "",encoding="utf-8")
        try:
            proc=subprocess.run([sys.executable,"-m","bandit","-q","-f","json",str(path)],capture_output=True,text=True,timeout=15)
            data=json.loads(proc.stdout or "{}")
            if proc.returncode not in (0,1) or not isinstance(data.get("results"),list): raise RuntimeError("invalid scan report")
            findings=[{"test_id":x.get("test_id"),"severity":x.get("issue_severity"),"confidence":x.get("issue_confidence"),"text":x.get("issue_text")} for x in data["results"]]
        except Exception as exc:
            return {"security_score":None,"security_status":"scanner_error","security_findings":json.dumps([{"severity":"UNKNOWN","text":type(exc).__name__}]),"security_issue_count":None}
    penalty=sum({"HIGH":25,"MEDIUM":10,"LOW":3}.get(x["severity"],0) for x in findings)
    return {"security_score":float(max(0,100-penalty)),"security_status":"completed","security_findings":json.dumps(findings,ensure_ascii=False),"security_issue_count":len(findings)}

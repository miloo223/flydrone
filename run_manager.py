# run_manager.py
from pathlib import Path
from datetime import datetime, timezone, timedelta
import yaml, git, shutil

KST = timezone(timedelta(hours=9))
RUNS_ROOT = Path("experiments")

def create_session(cfg: dict) -> Path:
    ts   = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    sess = (RUNS_ROOT / ts).resolve()     # ← 절대경로
    (sess / "logs").mkdir(parents=True)
    (sess / "plots" / "maps").mkdir(parents=True)
    (sess / "models").mkdir()
    (sess / "tensorboard").mkdir()

    # 설정·git 상태 저장
    (sess / "config.yaml").write_text(yaml.dump(cfg, sort_keys=False))
    try:
        repo = git.Repo(search_parent_directories=True)
        (sess / "git_commit.txt").write_text(repo.head.object.hexsha)
        (sess / "git_diff.patch").write_text(repo.git.diff())
    except git.InvalidGitRepositoryError:
        pass
    if Path("requirements.txt").exists():
        shutil.copy("requirements.txt", sess / "requirements.txt")

    print(f"▶ 세션 디렉터리: {sess}")   # ← 그냥 출력
    return sess

import git
import os
import logging


def get_git_commit_hash():
    try:
        repo = git.Repo(search_parent_directories=True)
        commit_hash = repo.head.object.hexsha
        return commit_hash
    except Exception as e:
        print(f"WARNING: Can't get git commit (maybe not a git repo?): {e}")
        return None


def setup_logging(log_dir, log_file):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
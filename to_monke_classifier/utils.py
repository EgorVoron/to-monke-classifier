import git
import logging

def get_git_commit_hash():
    try:
        repo = git.Repo(search_parent_directories=True)
        commit_hash = repo.head.object.hexsha
        return commit_hash
    except Exception as e:
        logging.error(f"git doesn't work: {e}")
        return None

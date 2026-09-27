from lingxi.skills.base import Outcome, Skill, SkillSet
from lingxi.skills.local import local_skills
from lingxi.skills.search import WebSearch
from lingxi.skills.web import web_skills


def builtin_skills() -> list[Skill]:
    return web_skills() + [WebSearch()] + local_skills()


__all__ = ["Outcome", "Skill", "SkillSet", "builtin_skills"]

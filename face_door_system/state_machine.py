from dataclasses import dataclass
from typing import Optional


class SystemState:
    WAITING = "WAITING"
    MATCHING = "MATCHING"
    CONFIRMED = "CONFIRMED"
    OPEN = "OPEN"
    COOLDOWN = "COOLDOWN"


@dataclass
class StateContext:
    current_person: Optional[str] = None
    score_count: int = 0
    cooldown_until: float = 0
    last_similarity: float = 0.0


class StateMachine:
    def __init__(self, logger=None):
        self.state = SystemState.WAITING
        self.context = StateContext()
        self.logger = logger

    def set_state(
        self,
        new_state: str,
        reason: str = "manual",
        similarity: Optional[float] = None,
        now: Optional[float] = None
    ) -> None:
        old_state = self.state
        self.state = new_state
        self.log_state_transition(
            old_state=old_state,
            new_state=new_state,
            reason=reason,
            similarity=similarity,
            now=now
        )

    def get_state(self) -> str:
        return self.state

    def log_state_transition(
        self,
        old_state: str,
        new_state: str,
        reason: str,
        similarity: Optional[float] = None,
        now: Optional[float] = None
    ) -> None:
        if not self.logger:
            return

        cooldown_remaining = self.get_cooldown_remaining(now)
        similarity_value = (
            self.context.last_similarity if similarity is None else similarity
        )
        self.logger.info(
            "STATE %s -> %s | reason=%s | person=%s | score_count=%s | "
            "similarity=%.3f | cooldown_remaining=%.3f",
            old_state,
            new_state,
            reason,
            self.context.current_person,
            self.context.score_count,
            similarity_value,
            cooldown_remaining
        )

    @property
    def current_person(self) -> Optional[str]:
        return self.context.current_person

    @property
    def score_count(self) -> int:
        return self.context.score_count

    @property
    def cooldown_until(self) -> float:
        return self.context.cooldown_until

    @property
    def last_similarity(self) -> float:
        return self.context.last_similarity

    def record_recognition(self, person_name: str, similarity: float) -> None:
        self.context.last_similarity = similarity

    def is_in_cooldown(self, now: float) -> bool:
        return now < self.context.cooldown_until

    def get_cooldown_remaining(self, now: Optional[float] = None) -> float:
        if now is None:
            import time

            now = time.time()
        return max(0.0, self.context.cooldown_until - now)

    def enter_cooldown(
        self,
        cooldown_until: float,
        reason: str = "open_success",
        now: Optional[float] = None
    ) -> None:
        self.context.cooldown_until = cooldown_until
        self.set_state(SystemState.COOLDOWN, reason=reason, now=now)

    def leave_cooldown(
        self,
        reason: str = "cooldown_expired",
        now: Optional[float] = None
    ) -> None:
        self.context.cooldown_until = 0
        self.set_state(SystemState.WAITING, reason=reason, now=now)

    def start_matching(
        self,
        person_name: str,
        similarity: Optional[float] = None,
        reason: str = "first_valid_match"
    ) -> None:
        self.context.current_person = person_name
        self.context.score_count = 1
        self.set_state(
            SystemState.MATCHING,
            reason=reason,
            similarity=similarity
        )

    def accumulate_matching(self, person_name: str) -> bool:
        if person_name == self.context.current_person:
            self.context.score_count += 1
            return True

        self.context.current_person = person_name
        self.context.score_count = 1
        return False

    def clear_match_context(self) -> None:
        self.context.current_person = None
        self.context.score_count = 0

    def reset_matching(
        self,
        reason: str = "match_reset",
        similarity: Optional[float] = None
    ) -> None:
        self.clear_match_context()
        self.set_state(
            SystemState.WAITING,
            reason=reason,
            similarity=similarity
        )

    def confirm(
        self,
        reason: str = "open_score_reached",
        similarity: Optional[float] = None
    ) -> None:
        self.set_state(
            SystemState.CONFIRMED,
            reason=reason,
            similarity=similarity
        )

    def open(
        self,
        reason: str = "access_confirmed",
        similarity: Optional[float] = None
    ) -> None:
        self.set_state(
            SystemState.OPEN,
            reason=reason,
            similarity=similarity
        )

    def get_status_snapshot(self, now: Optional[float] = None) -> dict:
        cooldown_remaining = self.get_cooldown_remaining(now)
        return {
            "state": self.state,
            "current_person": self.context.current_person,
            "score_count": self.context.score_count,
            "last_similarity": self.context.last_similarity,
            "in_cooldown": cooldown_remaining > 0,
            "cooldown_remaining": round(cooldown_remaining, 3),
        }

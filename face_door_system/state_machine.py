from dataclasses import dataclass
from datetime import datetime
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
    last_recognition_name: Optional[str] = None
    last_recognition_similarity: float = 0.0
    last_recognition_time: Optional[str] = None


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
        message = (
            "STATE %s -> %s | reason=%s | person=%s | score_count=%s | "
            "similarity=%.3f | cooldown_remaining=%.3f"
        )
        args = (
            old_state,
            new_state,
            reason,
            self.context.current_person,
            self.context.score_count,
            similarity_value,
            cooldown_remaining,
        )
        extra = {
            "event_type": "state_transition",
            "state_from": old_state,
            "state_to": new_state,
            "reason": reason,
            "person_name": self.context.current_person,
            "score_count": self.context.score_count,
            "similarity": round(similarity_value, 3),
            "cooldown_remaining": round(cooldown_remaining, 3),
        }

        try:
            self.logger.info(message, *args, extra=extra)
        except TypeError:
            self.logger.info(message, *args)

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
        self.context.last_recognition_name = person_name
        self.context.last_recognition_similarity = similarity
        self.context.last_recognition_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    def is_in_cooldown(self, now: float) -> bool:
        return now < self.context.cooldown_until

    def refresh_cooldown(
        self,
        now: float,
        reason: str = "cooldown_expired"
    ) -> bool:
        if self.state != SystemState.COOLDOWN:
            return False

        if self.is_in_cooldown(now):
            return False

        self.leave_cooldown(reason=reason, now=now)
        return True

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
            "recent_recognition": {
                "person_name": self.context.last_recognition_name,
                "similarity": round(self.context.last_recognition_similarity, 3),
                "timestamp": self.context.last_recognition_time,
            } if self.context.last_recognition_time else None,
            "in_cooldown": cooldown_remaining > 0,
            "cooldown_remaining": round(cooldown_remaining, 3),
        }

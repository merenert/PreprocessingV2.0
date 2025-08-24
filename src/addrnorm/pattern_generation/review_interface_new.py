"""
Pattern Review Interface

Human review interface for pattern suggestions and conflict resolutions.
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any, Set
from pathlib import Path
from collections import defaultdict

from .models import (
    PatternSuggestion,
    PatternConflict,
    PatternReview,
    ReviewStatus,
    ValidationStatus,
    ConflictResolution,
    ResolutionStrategy,
)

logger = logging.getLogger(__name__)


class PatternReviewInterface:
    """
    Human review interface for ML-generated patterns and conflict resolutions
    """

    def __init__(self, review_storage_path: Optional[str] = None):
        self.review_storage_path = Path(review_storage_path) if review_storage_path else Path("pattern_reviews")
        self.review_storage_path.mkdir(exist_ok=True)

        self.pending_reviews = {}
        self.completed_reviews = {}
        self.review_statistics = defaultdict(int)

        self._load_existing_reviews()

    def submit_for_review(
        self,
        suggestions: List[PatternSuggestion],
        conflicts: List[PatternConflict],
        reviewer_id: str = "default",
        priority: str = "normal",
    ) -> str:
        """
        Submit patterns and conflicts for human review

        Args:
            suggestions: Pattern suggestions to review
            conflicts: Detected conflicts to resolve
            reviewer_id: ID of the reviewer
            priority: Review priority (low, normal, high, critical)

        Returns:
            Review session ID
        """

        review_id = self._generate_review_id()

        review_session = {
            "review_id": review_id,
            "reviewer_id": reviewer_id,
            "priority": priority,
            "submitted_at": datetime.now().isoformat(),
            "status": ReviewStatus.PENDING.value,
            "pattern_suggestions": [self._serialize_suggestion(s) for s in suggestions],
            "conflicts": [self._serialize_conflict(c) for c in conflicts],
            "review_decisions": {},
            "conflict_resolutions": {},
            "reviewer_notes": "",
            "completion_time": None,
        }

        self.pending_reviews[review_id] = review_session
        self._save_review_session(review_session)

        logger.info(f"Submitted review session {review_id} with {len(suggestions)} patterns and {len(conflicts)} conflicts")

        return review_id

    def get_pending_reviews(self, reviewer_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of pending reviews"""

        pending = []

        for review_id, session in self.pending_reviews.items():
            if reviewer_id and session["reviewer_id"] != reviewer_id:
                continue

            summary = {
                "review_id": review_id,
                "reviewer_id": session["reviewer_id"],
                "priority": session["priority"],
                "submitted_at": session["submitted_at"],
                "pattern_count": len(session["pattern_suggestions"]),
                "conflict_count": len(session["conflicts"]),
                "estimated_time": self._estimate_review_time(session),
            }
            pending.append(summary)

        # Sort by priority and submission time
        priority_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
        pending.sort(key=lambda x: (priority_order.get(x["priority"], 2), x["submitted_at"]))

        return pending

    def get_review_session(self, review_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed review session information"""

        if review_id in self.pending_reviews:
            session = self.pending_reviews[review_id].copy()
            session["review_summary"] = self._generate_review_summary(session)
            return session

        elif review_id in self.completed_reviews:
            return self.completed_reviews[review_id]

        else:
            # Try to load from storage
            return self._load_review_session(review_id)

    def review_pattern(
        self, review_id: str, pattern_id: str, decision: str, notes: str = "", modifications: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Submit review decision for a specific pattern

        Args:
            review_id: Review session ID
            pattern_id: Pattern ID being reviewed
            decision: Review decision (approve, reject, modify)
            notes: Reviewer notes
            modifications: Suggested modifications if decision is 'modify'

        Returns:
            Success status
        """

        if review_id not in self.pending_reviews:
            logger.error(f"Review session {review_id} not found")
            return False

        session = self.pending_reviews[review_id]

        # Find the pattern
        pattern_found = False
        for pattern in session["pattern_suggestions"]:
            if pattern["pattern_id"] == pattern_id:
                pattern_found = True
                break

        if not pattern_found:
            logger.error(f"Pattern {pattern_id} not found in review session {review_id}")
            return False

        # Record decision
        decision_record = {
            "decision": decision,
            "notes": notes,
            "reviewed_at": datetime.now().isoformat(),
            "modifications": modifications or {},
        }

        session["review_decisions"][pattern_id] = decision_record

        # Update statistics
        self.review_statistics[f"pattern_{decision}"] += 1

        self._save_review_session(session)

        logger.info(f"Recorded {decision} decision for pattern {pattern_id} in review {review_id}")

        return True

    def resolve_conflict(
        self,
        review_id: str,
        conflict_id: str,
        chosen_resolution: ResolutionStrategy,
        notes: str = "",
        custom_resolution: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Submit conflict resolution decision

        Args:
            review_id: Review session ID
            conflict_id: Conflict ID being resolved
            chosen_resolution: Chosen resolution strategy
            notes: Reviewer notes
            custom_resolution: Custom resolution details if applicable

        Returns:
            Success status
        """

        if review_id not in self.pending_reviews:
            logger.error(f"Review session {review_id} not found")
            return False

        session = self.pending_reviews[review_id]

        # Find the conflict
        conflict_found = False
        for conflict in session["conflicts"]:
            if conflict["conflict_id"] == conflict_id:
                conflict_found = True
                break

        if not conflict_found:
            logger.error(f"Conflict {conflict_id} not found in review session {review_id}")
            return False

        # Record resolution
        resolution_record = {
            "chosen_strategy": chosen_resolution.value,
            "notes": notes,
            "resolved_at": datetime.now().isoformat(),
            "custom_resolution": custom_resolution or {},
        }

        session["conflict_resolutions"][conflict_id] = resolution_record

        # Update statistics
        self.review_statistics[f"conflict_{chosen_resolution.value}"] += 1

        self._save_review_session(session)

        logger.info(f"Recorded {chosen_resolution.value} resolution for conflict {conflict_id} in review {review_id}")

        return True

    def complete_review(self, review_id: str, overall_notes: str = "") -> bool:
        """
        Mark review session as complete

        Args:
            review_id: Review session ID
            overall_notes: Overall reviewer notes

        Returns:
            Success status
        """

        if review_id not in self.pending_reviews:
            logger.error(f"Review session {review_id} not found")
            return False

        session = self.pending_reviews[review_id]

        # Check if all items have been reviewed
        pattern_count = len(session["pattern_suggestions"])
        conflict_count = len(session["conflicts"])

        reviewed_patterns = len(session["review_decisions"])
        resolved_conflicts = len(session["conflict_resolutions"])

        if reviewed_patterns < pattern_count or resolved_conflicts < conflict_count:
            logger.warning(
                f"Review {review_id} incomplete: {reviewed_patterns}/{pattern_count} patterns, {resolved_conflicts}/{conflict_count} conflicts"
            )
            return False

        # Mark as complete
        session["status"] = ReviewStatus.COMPLETED.value
        session["reviewer_notes"] = overall_notes
        session["completion_time"] = datetime.now().isoformat()

        # Move to completed reviews
        self.completed_reviews[review_id] = session
        del self.pending_reviews[review_id]

        # Update statistics
        self.review_statistics["completed_sessions"] += 1

        self._save_review_session(session)

        logger.info(f"Completed review session {review_id}")

        return True

    def get_review_statistics(self) -> Dict[str, Any]:
        """Get review statistics and metrics"""

        total_pending = len(self.pending_reviews)
        total_completed = len(self.completed_reviews)

        # Calculate average review times
        completion_times = []
        for session in self.completed_reviews.values():
            if session.get("completion_time") and session.get("submitted_at"):
                submitted = datetime.fromisoformat(session["submitted_at"])
                completed = datetime.fromisoformat(session["completion_time"])
                duration = (completed - submitted).total_seconds() / 3600  # hours
                completion_times.append(duration)

        avg_review_time = sum(completion_times) / len(completion_times) if completion_times else 0

        # Pattern decision breakdown
        pattern_decisions = {
            "approved": self.review_statistics.get("pattern_approve", 0),
            "rejected": self.review_statistics.get("pattern_reject", 0),
            "modified": self.review_statistics.get("pattern_modify", 0),
        }

        # Conflict resolution breakdown
        conflict_resolutions = {}
        for strategy in ResolutionStrategy:
            count = self.review_statistics.get(f"conflict_{strategy.value}", 0)
            if count > 0:
                conflict_resolutions[strategy.value] = count

        return {
            "review_sessions": {
                "pending": total_pending,
                "completed": total_completed,
                "total": total_pending + total_completed,
            },
            "average_review_time_hours": round(avg_review_time, 2),
            "pattern_decisions": pattern_decisions,
            "conflict_resolutions": conflict_resolutions,
            "last_updated": datetime.now().isoformat(),
        }

    def export_review_data(self, output_path: str) -> bool:
        """Export review data for analysis"""

        try:
            export_data = {
                "pending_reviews": self.pending_reviews,
                "completed_reviews": self.completed_reviews,
                "statistics": self.get_review_statistics(),
                "exported_at": datetime.now().isoformat(),
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported review data to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export review data: {e}")
            return False

    def _serialize_suggestion(self, suggestion: PatternSuggestion) -> Dict[str, Any]:
        """Serialize PatternSuggestion for storage"""

        return {
            "pattern_id": suggestion.pattern_id,
            "regex_pattern": suggestion.regex_pattern,
            "description": suggestion.description,
            "confidence_score": suggestion.confidence_score,
            "expected_fields": suggestion.expected_fields,
            "sample_matches": suggestion.sample_matches,
            "quality_metrics": suggestion.quality_metrics,
            "generated_at": suggestion.generated_at.isoformat(),
        }

    def _serialize_conflict(self, conflict: PatternConflict) -> Dict[str, Any]:
        """Serialize PatternConflict for storage"""

        return {
            "conflict_id": conflict.conflict_id,
            "existing_pattern_id": conflict.existing_pattern_id,
            "new_pattern_id": conflict.new_pattern_id,
            "conflict_type": conflict.conflict_type.value,
            "severity": conflict.severity.value,
            "description": conflict.description,
            "affected_samples": conflict.affected_samples,
            "resolution_suggestions": [
                {
                    "strategy": res.strategy.value,
                    "description": res.description,
                    "confidence_score": res.confidence_score,
                    "implementation_notes": res.implementation_notes,
                    "impact_assessment": res.impact_assessment,
                }
                for res in conflict.resolution_suggestions
            ],
            "conflict_details": conflict.conflict_details,
            "detected_at": conflict.detected_at.isoformat(),
        }

    def _generate_review_summary(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary information for review session"""

        patterns = session["pattern_suggestions"]
        conflicts = session["conflicts"]

        # Pattern analysis
        avg_confidence = sum(p["confidence_score"] for p in patterns) / len(patterns) if patterns else 0
        high_confidence_patterns = sum(1 for p in patterns if p["confidence_score"] > 0.8)

        # Conflict analysis
        high_severity_conflicts = sum(1 for c in conflicts if c["severity"] in ["HIGH", "CRITICAL"])
        conflict_types = list(set(c["conflict_type"] for c in conflicts))

        return {
            "pattern_summary": {
                "total_patterns": len(patterns),
                "average_confidence": round(avg_confidence, 3),
                "high_confidence_count": high_confidence_patterns,
            },
            "conflict_summary": {
                "total_conflicts": len(conflicts),
                "high_severity_count": high_severity_conflicts,
                "conflict_types": conflict_types,
            },
            "review_progress": {
                "patterns_reviewed": len(session.get("review_decisions", {})),
                "conflicts_resolved": len(session.get("conflict_resolutions", {})),
            },
        }

    def _estimate_review_time(self, session: Dict[str, Any]) -> str:
        """Estimate time required for review"""

        pattern_count = len(session["pattern_suggestions"])
        conflict_count = len(session["conflicts"])

        # Rough estimates in minutes
        pattern_time = pattern_count * 3  # 3 minutes per pattern
        conflict_time = conflict_count * 5  # 5 minutes per conflict

        total_minutes = pattern_time + conflict_time

        if total_minutes < 60:
            return f"{total_minutes} minutes"
        else:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            return f"{hours}h {minutes}m"

    def _generate_review_id(self) -> str:
        """Generate unique review ID"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"review_{timestamp}"

    def _save_review_session(self, session: Dict[str, Any]) -> None:
        """Save review session to storage"""

        review_id = session["review_id"]
        file_path = self.review_storage_path / f"{review_id}.json"

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(session, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save review session {review_id}: {e}")

    def _load_review_session(self, review_id: str) -> Optional[Dict[str, Any]]:
        """Load review session from storage"""

        file_path = self.review_storage_path / f"{review_id}.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load review session {review_id}: {e}")
            return None

    def _load_existing_reviews(self) -> None:
        """Load existing reviews from storage"""

        if not self.review_storage_path.exists():
            return

        for file_path in self.review_storage_path.glob("review_*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    session = json.load(f)

                review_id = session["review_id"]
                status = session.get("status", "PENDING")

                if status == ReviewStatus.COMPLETED.value:
                    self.completed_reviews[review_id] = session
                else:
                    self.pending_reviews[review_id] = session

            except Exception as e:
                logger.error(f"Failed to load review from {file_path}: {e}")
                continue

        logger.info(f"Loaded {len(self.pending_reviews)} pending and {len(self.completed_reviews)} completed reviews")


class ReviewCLI:
    """
    Command-line interface for pattern review
    """

    def __init__(self, review_interface: PatternReviewInterface):
        self.review_interface = review_interface

    def show_pending_reviews(self) -> None:
        """Show pending reviews"""

        pending = self.review_interface.get_pending_reviews()

        if not pending:
            print("No pending reviews.")
            return

        print(f"\n📋 Pending Reviews ({len(pending)} total)")
        print("=" * 60)

        for review in pending:
            priority_emoji = {"critical": "🔴", "high": "🟠", "normal": "🟡", "low": "🟢"}.get(review["priority"], "⚪")

            print(f"{priority_emoji} {review['review_id']}")
            print(f"   Reviewer: {review['reviewer_id']}")
            print(f"   Patterns: {review['pattern_count']}, Conflicts: {review['conflict_count']}")
            print(f"   Estimated time: {review['estimated_time']}")
            print(f"   Submitted: {review['submitted_at']}")
            print()

    def show_review_session(self, review_id: str) -> None:
        """Show detailed review session"""

        session = self.review_interface.get_review_session(review_id)

        if not session:
            print(f"Review session {review_id} not found.")
            return

        print(f"\n📄 Review Session: {review_id}")
        print("=" * 60)

        summary = session.get("review_summary", {})

        # Pattern summary
        if "pattern_summary" in summary:
            ps = summary["pattern_summary"]
            print(f"📊 Patterns: {ps['total_patterns']} total")
            print(f"   Average confidence: {ps['average_confidence']:.3f}")
            print(f"   High confidence: {ps['high_confidence_count']}")

        # Conflict summary
        if "conflict_summary" in summary:
            cs = summary["conflict_summary"]
            print(f"⚠️  Conflicts: {cs['total_conflicts']} total")
            print(f"   High severity: {cs['high_severity_count']}")
            print(f"   Types: {', '.join(cs['conflict_types'])}")

        # Progress
        if "review_progress" in summary:
            rp = summary["review_progress"]
            print(f"✅ Progress:")
            print(f"   Patterns reviewed: {rp['patterns_reviewed']}")
            print(f"   Conflicts resolved: {rp['conflicts_resolved']}")

        print()

    def show_statistics(self) -> None:
        """Show review statistics"""

        stats = self.review_interface.get_review_statistics()

        print("\n📈 Review Statistics")
        print("=" * 40)

        # Sessions
        sessions = stats["review_sessions"]
        print(f"Sessions: {sessions['total']} total")
        print(f"  Pending: {sessions['pending']}")
        print(f"  Completed: {sessions['completed']}")

        # Time
        if stats["average_review_time_hours"] > 0:
            print(f"Average review time: {stats['average_review_time_hours']:.1f} hours")

        # Pattern decisions
        if stats["pattern_decisions"]:
            print("\nPattern Decisions:")
            for decision, count in stats["pattern_decisions"].items():
                if count > 0:
                    print(f"  {decision.capitalize()}: {count}")

        # Conflict resolutions
        if stats["conflict_resolutions"]:
            print("\nConflict Resolutions:")
            for resolution, count in stats["conflict_resolutions"].items():
                print(f"  {resolution}: {count}")

        print()

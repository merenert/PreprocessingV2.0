"""
ML Pattern Suggester

Advanced ML-based pattern suggestion system with clustering and sequence analysis.
"""

import re
import logging
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import Counter, defaultdict
import unicodedata
import hashlib

from .models import PatternSuggestion, AddressCluster, PatternGenerationConfig, ConflictType, PatternConflict

logger = logging.getLogger(__name__)


class AddressFeatureExtractor:
    """Extract features from addresses for clustering"""

    def __init__(self):
        self.turkish_pattern_keywords = [
            "mahalle",
            "mah",
            "sokak",
            "sok",
            "cadde",
            "cad",
            "bulvar",
            "blv",
            "no",
            "numara",
            "daire",
            "kat",
            "blok",
            "site",
            "apt",
            "apartman",
        ]

    def extract_features(self, addresses: List[str]) -> Dict[str, Any]:
        """Extract comprehensive features from address list"""

        features = {
            "length_stats": self._extract_length_features(addresses),
            "token_stats": self._extract_token_features(addresses),
            "pattern_stats": self._extract_pattern_features(addresses),
            "structure_stats": self._extract_structure_features(addresses),
            "numeric_stats": self._extract_numeric_features(addresses),
            "keyword_stats": self._extract_keyword_features(addresses),
        }

        return features

    def _extract_length_features(self, addresses: List[str]) -> Dict[str, float]:
        """Extract length-based features"""
        lengths = [len(addr) for addr in addresses]
        return {
            "avg_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "length_cv": np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0,
        }

    def _extract_token_features(self, addresses: List[str]) -> Dict[str, float]:
        """Extract token-based features"""
        token_counts = [len(addr.split()) for addr in addresses]
        return {
            "avg_tokens": np.mean(token_counts),
            "std_tokens": np.std(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
        }

    def _extract_pattern_features(self, addresses: List[str]) -> Dict[str, float]:
        """Extract pattern-based features"""

        # Character type patterns
        digit_ratios = [sum(c.isdigit() for c in addr) / len(addr) for addr in addresses if addr]
        alpha_ratios = [sum(c.isalpha() for c in addr) / len(addr) for addr in addresses if addr]
        space_ratios = [addr.count(" ") / len(addr) for addr in addresses if addr]

        return {
            "avg_digit_ratio": np.mean(digit_ratios) if digit_ratios else 0,
            "avg_alpha_ratio": np.mean(alpha_ratios) if alpha_ratios else 0,
            "avg_space_ratio": np.mean(space_ratios) if space_ratios else 0,
            "digit_ratio_std": np.std(digit_ratios) if digit_ratios else 0,
        }

    def _extract_structure_features(self, addresses: List[str]) -> Dict[str, float]:
        """Extract structural features"""

        # Common separators
        separators = [",", "/", "-", ":", "."]
        separator_counts = {}

        for sep in separators:
            counts = [addr.count(sep) for addr in addresses]
            separator_counts[f"{sep}_avg"] = np.mean(counts)
            separator_counts[f"{sep}_std"] = np.std(counts)

        return separator_counts

    def _extract_numeric_features(self, addresses: List[str]) -> Dict[str, float]:
        """Extract numeric pattern features"""

        # Find numeric patterns
        number_patterns = [
            r"\d+",  # Any number
            r"\d{5}",  # 5-digit (postal code)
            r"\d{1,3}",  # 1-3 digits (building number)
        ]

        pattern_stats = {}
        for i, pattern in enumerate(number_patterns):
            matches_per_addr = [len(re.findall(pattern, addr)) for addr in addresses]
            pattern_stats[f"numeric_pattern_{i}_avg"] = np.mean(matches_per_addr)
            pattern_stats[f"numeric_pattern_{i}_std"] = np.std(matches_per_addr)

        return pattern_stats

    def _extract_keyword_features(self, addresses: List[str]) -> Dict[str, float]:
        """Extract Turkish address keyword features"""

        keyword_stats = {}

        for keyword in self.turkish_pattern_keywords:
            occurrences = [addr.lower().count(keyword.lower()) for addr in addresses]
            keyword_stats[f"keyword_{keyword}_ratio"] = sum(occurrences) / len(addresses)
            keyword_stats[f"keyword_{keyword}_presence"] = sum(1 for c in occurrences if c > 0) / len(addresses)

        return keyword_stats


class MLPatternSuggester:
    """
    ML-based pattern suggestion system with advanced clustering and template generation
    """

    def __init__(self, config: Optional[PatternGenerationConfig] = None):
        self.config = config or PatternGenerationConfig()
        self.feature_extractor = AddressFeatureExtractor()
        self.vectorizers = {}
        self.clustering_models = {}
        self.pattern_cache = {}

    def suggest_patterns(self, addresses: List[str], max_patterns: int = 10) -> List[PatternSuggestion]:
        """
        Generate pattern suggestions from address list

        Args:
            addresses: List of address strings
            max_patterns: Maximum number of patterns to generate

        Returns:
            List of PatternSuggestion objects
        """

        logger.info(f"Generating patterns for {len(addresses)} addresses")

        # Step 1: Cluster addresses
        clusters = self._cluster_addresses(addresses)

        # Step 2: Extract patterns from clusters
        pattern_suggestions = []

        for cluster in clusters[:max_patterns]:
            try:
                pattern = self._extract_pattern_from_cluster(cluster)
                if pattern:
                    pattern_suggestions.append(pattern)
            except Exception as e:
                logger.error(f"Error extracting pattern from cluster {cluster.cluster_id}: {e}")
                continue

        # Step 3: Rank and filter patterns
        ranked_patterns = self._rank_patterns(pattern_suggestions)

        # Step 4: Apply quality threshold
        quality_filtered = [p for p in ranked_patterns if p.confidence_score >= self.config.min_pattern_confidence]

        logger.info(f"Generated {len(quality_filtered)} quality patterns from {len(pattern_suggestions)} candidates")

        return quality_filtered[:max_patterns]

    def _cluster_addresses(self, addresses: List[str]) -> List[AddressCluster]:
        """Cluster addresses by similarity"""

        if len(addresses) < self.config.min_cluster_size:
            # Create single cluster if too few addresses
            return [
                AddressCluster(
                    cluster_id="cluster_0",
                    addresses=addresses,
                    common_features=self.feature_extractor.extract_features(addresses),
                    cluster_size=len(addresses),
                    confidence_score=0.5,
                    representative_address=addresses[0] if addresses else "",
                )
            ]

        # Create feature vectors
        feature_matrix = self._create_feature_matrix(addresses)

        # Try different clustering methods
        best_clusters = None
        best_score = -1

        clustering_methods = [
            ("kmeans", self._kmeans_clustering),
            ("dbscan", self._dbscan_clustering),
            ("gaussian", self._gaussian_clustering),
        ]

        for method_name, clustering_func in clustering_methods:
            try:
                clusters = clustering_func(feature_matrix, addresses)
                score = self._evaluate_clustering(clusters)

                if score > best_score:
                    best_score = score
                    best_clusters = clusters

            except Exception as e:
                logger.warning(f"Clustering method {method_name} failed: {e}")
                continue

        return best_clusters or self._fallback_clustering(addresses)

    def _create_feature_matrix(self, addresses: List[str]) -> np.ndarray:
        """Create feature matrix for clustering"""

        # Text features using TF-IDF
        if "tfidf" not in self.vectorizers:
            self.vectorizers["tfidf"] = TfidfVectorizer(max_features=100, ngram_range=(1, 2), analyzer="word")

        # Character n-gram features
        if "char_ngram" not in self.vectorizers:
            self.vectorizers["char_ngram"] = TfidfVectorizer(max_features=50, ngram_range=(2, 4), analyzer="char")

        # Fit and transform
        tfidf_features = self.vectorizers["tfidf"].fit_transform(addresses).toarray()
        char_features = self.vectorizers["char_ngram"].fit_transform(addresses).toarray()

        # Combine feature matrices
        combined_features = np.hstack([tfidf_features, char_features])

        # Add manual features
        manual_features = self._extract_manual_features(addresses)

        if manual_features.shape[1] > 0:
            combined_features = np.hstack([combined_features, manual_features])

        return combined_features

    def _extract_manual_features(self, addresses: List[str]) -> np.ndarray:
        """Extract manual engineered features"""

        features = []

        for addr in addresses:
            addr_features = [
                len(addr),  # Length
                addr.count(" "),  # Number of spaces
                len(re.findall(r"\d+", addr)),  # Number of numeric tokens
                addr.count(","),  # Commas
                addr.count("/"),  # Slashes
                sum(1 for c in addr if c.isupper()),  # Uppercase count
                sum(1 for c in addr if c.islower()),  # Lowercase count
                1 if "mahalle" in addr.lower() else 0,  # Has mahalle
                1 if "sokak" in addr.lower() else 0,  # Has sokak
                1 if re.search(r"\d{5}", addr) else 0,  # Has postal code pattern
            ]
            features.append(addr_features)

        return np.array(features)

    def _kmeans_clustering(self, feature_matrix: np.ndarray, addresses: List[str]) -> List[AddressCluster]:
        """K-means clustering"""

        # Determine optimal number of clusters
        max_clusters = min(self.config.max_clusters, len(addresses) // self.config.min_cluster_size)
        n_clusters = self._find_optimal_clusters(feature_matrix, max_clusters)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)

        return self._create_clusters_from_labels(cluster_labels, addresses)

    def _dbscan_clustering(self, feature_matrix: np.ndarray, addresses: List[str]) -> List[AddressCluster]:
        """DBSCAN clustering"""

        dbscan = DBSCAN(eps=0.5, min_samples=self.config.min_cluster_size)
        cluster_labels = dbscan.fit_predict(feature_matrix)

        return self._create_clusters_from_labels(cluster_labels, addresses)

    def _gaussian_clustering(self, feature_matrix: np.ndarray, addresses: List[str]) -> List[AddressCluster]:
        """Gaussian Mixture clustering"""

        max_clusters = min(self.config.max_clusters, len(addresses) // self.config.min_cluster_size)
        n_clusters = self._find_optimal_clusters(feature_matrix, max_clusters)

        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        cluster_labels = gmm.fit_predict(feature_matrix)

        return self._create_clusters_from_labels(cluster_labels, addresses)

    def _find_optimal_clusters(self, feature_matrix: np.ndarray, max_clusters: int) -> int:
        """Find optimal number of clusters using silhouette score"""

        if max_clusters <= 1:
            return 1

        best_score = -1
        best_k = 2

        for k in range(2, min(max_clusters + 1, 11)):  # Test up to 10 clusters
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(feature_matrix)

                # Skip if any cluster has too few samples
                if min(Counter(cluster_labels).values()) < self.config.min_cluster_size:
                    continue

                score = silhouette_score(feature_matrix, cluster_labels)

                if score > best_score:
                    best_score = score
                    best_k = k

            except Exception:
                continue

        return best_k

    def _create_clusters_from_labels(self, cluster_labels: np.ndarray, addresses: List[str]) -> List[AddressCluster]:
        """Create AddressCluster objects from cluster labels"""

        clusters = []
        unique_labels = set(cluster_labels)

        for label in unique_labels:
            if label == -1:  # DBSCAN noise points
                continue

            cluster_addresses = [addr for i, addr in enumerate(addresses) if cluster_labels[i] == label]

            if len(cluster_addresses) < self.config.min_cluster_size:
                continue

            # Extract features for this cluster
            cluster_features = self.feature_extractor.extract_features(cluster_addresses)

            # Calculate confidence based on cluster homogeneity
            confidence = self._calculate_cluster_confidence(cluster_addresses)

            # Find representative address (most typical)
            representative = self._find_representative_address(cluster_addresses)

            cluster = AddressCluster(
                cluster_id=f"cluster_{label}",
                addresses=cluster_addresses,
                common_features=cluster_features,
                cluster_size=len(cluster_addresses),
                confidence_score=confidence,
                representative_address=representative,
            )

            clusters.append(cluster)

        return sorted(clusters, key=lambda x: x.confidence_score, reverse=True)

    def _calculate_cluster_confidence(self, addresses: List[str]) -> float:
        """Calculate confidence score for a cluster"""

        # Similarity-based confidence
        similarities = []

        for i in range(len(addresses)):
            for j in range(i + 1, len(addresses)):
                sim = self._calculate_address_similarity(addresses[i], addresses[j])
                similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 0.0

        # Size-based confidence boost
        size_factor = min(1.0, len(addresses) / 20)

        # Combined confidence
        confidence = (avg_similarity * 0.7) + (size_factor * 0.3)

        return round(confidence, 3)

    def _calculate_address_similarity(self, addr1: str, addr2: str) -> float:
        """Calculate similarity between two addresses"""

        # Jaccard similarity on tokens
        tokens1 = set(addr1.lower().split())
        tokens2 = set(addr2.lower().split())

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        jaccard = len(intersection) / len(union) if union else 0.0

        # Character-level similarity
        char_sim = 1.0 - (abs(len(addr1) - len(addr2)) / max(len(addr1), len(addr2)))

        # Combined similarity
        return (jaccard * 0.7) + (char_sim * 0.3)

    def _find_representative_address(self, addresses: List[str]) -> str:
        """Find most representative address in cluster"""

        if len(addresses) == 1:
            return addresses[0]

        # Calculate average similarity to all other addresses
        avg_similarities = []

        for addr in addresses:
            similarities = [self._calculate_address_similarity(addr, other) for other in addresses if other != addr]
            avg_similarities.append(np.mean(similarities))

        # Return address with highest average similarity
        best_idx = np.argmax(avg_similarities)
        return addresses[best_idx]

    def _extract_pattern_from_cluster(self, cluster: AddressCluster) -> Optional[PatternSuggestion]:
        """Extract regex pattern from address cluster"""

        addresses = cluster.addresses

        # Simple pattern extraction - create basic regex
        # This is a simplified version for demo purposes
        pattern_parts = []

        # Look for common elements
        if any("mahalle" in addr.lower() for addr in addresses):
            pattern_parts.append(r"(?P<mahalle>\w+)\s*(mahalle|mah)\.?")

        if any("sokak" in addr.lower() for addr in addresses):
            pattern_parts.append(r"(?P<sokak>\w+)\s*(sokak|sok)\.?")

        if any("cadde" in addr.lower() for addr in addresses):
            pattern_parts.append(r"(?P<cadde>\w+)\s*(cadde|cad)\.?")

        # Number patterns
        if any(re.search(r"no:?\s*\d+", addr.lower()) for addr in addresses):
            pattern_parts.append(r"no:?\s*(?P<no>\d+)")

        if not pattern_parts:
            pattern_parts = [r"\w+"]  # Fallback pattern

        # Join with flexible separators
        regex_pattern = r"\s*[,\-\s]*\s*".join(pattern_parts)

        # Generate pattern ID
        pattern_id = self._generate_pattern_id(regex_pattern)

        return PatternSuggestion(
            pattern_id=pattern_id,
            regex_pattern=regex_pattern,
            description=f"Auto-generated pattern from cluster {cluster.cluster_id}",
            source_cluster=cluster,
            confidence_score=cluster.confidence_score,
            expected_fields=[],  # Simplified for demo
            sample_matches=addresses[:3],  # First 3 as samples
            quality_metrics={},  # Simplified for demo
            generated_at=datetime.now(),
        )

    def _rank_patterns(self, patterns: List[PatternSuggestion]) -> List[PatternSuggestion]:
        """Rank patterns by quality score"""

        return sorted(patterns, key=lambda p: p.confidence_score, reverse=True)

    def _evaluate_clustering(self, clusters: List[AddressCluster]) -> float:
        """Evaluate clustering quality"""

        if not clusters:
            return 0.0

        # Metrics: cluster count, average size, confidence
        cluster_count = len(clusters)
        avg_size = np.mean([c.cluster_size for c in clusters])
        avg_confidence = np.mean([c.confidence_score for c in clusters])

        # Penalize too many or too few clusters
        size_penalty = 1.0
        if cluster_count > self.config.max_clusters:
            size_penalty = 0.5
        elif cluster_count < 2:
            size_penalty = 0.7

        score = avg_confidence * size_penalty
        return score

    def _fallback_clustering(self, addresses: List[str]) -> List[AddressCluster]:
        """Fallback clustering when ML methods fail"""

        # Simple length-based clustering
        length_groups = defaultdict(list)

        for addr in addresses:
            length_bucket = len(addr) // 20  # Group by length buckets
            length_groups[length_bucket].append(addr)

        clusters = []
        for i, (bucket, group_addresses) in enumerate(length_groups.items()):
            if len(group_addresses) >= self.config.min_cluster_size:
                cluster = AddressCluster(
                    cluster_id=f"fallback_{i}",
                    addresses=group_addresses,
                    common_features={"length_bucket": bucket},
                    cluster_size=len(group_addresses),
                    confidence_score=0.3,  # Low confidence for fallback
                    representative_address=group_addresses[0],
                )
                clusters.append(cluster)

        return clusters

    def _generate_pattern_id(self, pattern: str) -> str:
        """Generate unique pattern ID"""
        return f"ml_pattern_{hashlib.md5(pattern.encode()).hexdigest()[:8]}"

"""
queries.py

This module provides functions to fetch claims and related premise data with an optional local cache layer.
If the cache file exists and the DB has no new rows, the results are loaded from cache. Otherwise, fresh data
is fetched from the DB and dumped into the cache for subsequent runs.
Cache behavior is controlled by the `CACHE_ENABLED` flag in the project's `config.py`.
"""
import pickle
from pathlib import Path
from .db import get_session
from .models import ADU, Relationship
from .config import CACHE_ENABLED
from collections import defaultdict
from .quality_data import data
# --- Cache configuration ---
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
CLAIMS_CACHE_FILE = CACHE_DIR / 'claims_cache.pkl'
DATA_CACHE_FILE   = CACHE_DIR / 'data_cache.pkl'

# --- Internal helpers ---

def _make_claims_query(session):
    """Return a base query for all claims ordered by ID."""
    return (
        session
        .query(ADU)
        .filter(ADU.type == 'claim')
        .order_by(ADU.id)
    )


def _get_claims(session, split: str = 'training') -> list[ADU]:
    """
    Get claims split into 'training', 'test', or 'benchmark'.
    """
    q = _make_claims_query(session)
    total = q.count()
    training_split = int(total * 0.7)
    test_split = int(total * 0.9)

    if split == 'training':
        return q.limit(training_split).all()
    elif split == 'test':
        return q.offset(training_split).limit(test_split - training_split).all()
    elif split == 'benchmark':
        return q.offset(test_split).all()
    else:
        raise ValueError("Invalid split. Must be 'training', 'test', or 'benchmark'")


def _get_data(session, split: str = 'training') -> tuple[list[ADU], list[ADU], list[str]]:
    """Get tuples of (claims, premises, relationship_category (stance_pro/stance_con) )."""

    # Fetch claims
    initial_claims = _get_claims(session, split)
    ids = [c.id for c in initial_claims]

    # Fetch related premises
    rows = (
        session
        .query(ADU, Relationship.category, Relationship.to_adu_id)
        .join(Relationship, Relationship.from_adu_id == ADU.id)
        .filter(ADU.type == 'premise', Relationship.to_adu_id.in_(ids))
        .all()
    )

    # If rows is empty, it means no premises were found for any of the initial_claims.
    # However, we need to group to see which specific claims have premises.
    premises, categories, claim_ids = zip(*rows) if rows else ([], [], [])

    # Group premises by claim_id
    grouped_premises_by_claim_id = defaultdict(list)
    for p, cat, cid in zip(premises, categories, claim_ids):
        grouped_premises_by_claim_id[cid].append((p, cat))


    # Prepare final lists for output
    output_claims = []
    output_premises = []
    output_categories = []

    # Alternate stances
    relationship_options = ['stance_pro', 'stance_con']
    
    # Iterate through the initially fetched claims and filter those without premises
    for claim_obj in initial_claims:
        if claim_obj.id in grouped_premises_by_claim_id:
            # This claim has premises, so include it
            output_claims.append(claim_obj)

            # Determine the desired stance based on the count of *included* claims.
            # Original code's enumerate(claims, start=1) means idx % 2 gives:
            # idx=1 (1st item) -> 1%2=1 (e.g., stance_con if options are [pro, con])
            # idx=2 (2nd item) -> 2%2=0 (e.g., stance_pro)
            # We use len(output_claims) which is 1-based for the current count.
            desired_stance_idx = len(output_claims) % 2 
            desired_stance = relationship_options[desired_stance_idx]

            # Get all premises for this specific claim
            current_claim_premises_list = grouped_premises_by_claim_id[claim_obj.id]
            
            # Try to find a premise with the desired stance
            chosen_premise_tuple = next(
                ((p, cat) for p, cat in current_claim_premises_list if cat == desired_stance),
                None
            )

            if chosen_premise_tuple is None:
                # If the desired stance wasn't found, pick the first available premise for this claim.
                # We know current_claim_premises_list is not empty because claim_obj.id is in grouped_premises_by_claim_id.
                chosen_premise_tuple = current_claim_premises_list[0]
            
            output_premises.append(chosen_premise_tuple[0])   # The ADU object of the premise
            output_categories.append(chosen_premise_tuple[1]) # The category string
            
    return output_claims, output_premises, output_categories


def _load_cache(file_path: Path):
    """Load cached data from `file_path`, or return None if missing/corrupt."""
    try:
        if file_path.exists():
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    except Exception:
        pass
    return None


def _save_cache(file_path: Path, data):
    """Dump `data` to `file_path` via pickle."""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

# --- Public API ---

def get_training_claims() -> list[ADU]:
    """Return the first 70% of claims, using cache if enabled and up-to-date."""
    with get_session() as session:
        if not CACHE_ENABLED:
            return _get_claims(session, training=True)
        total = session.query(ADU).filter(ADU.type == 'claim').count()
        cache = _load_cache(CLAIMS_CACHE_FILE)
        if cache and cache.get('total') == total:
            return cache['training']
        training = _get_claims(session, split='training')
        test_ = _get_claims(session, split='test')
        benchmark = _get_claims(session, split='benchmark')
        _save_cache(CLAIMS_CACHE_FILE, {'total': total, 'training': training, 'test': test_, 'benchmark': benchmark})
        return training


def get_test_claims() -> list[ADU]:
    """Return the second 20% of claims, using cache if enabled and up-to-date."""
    with get_session() as session:
        if not CACHE_ENABLED:
            return _get_claims(session, training=False)
        total = session.query(ADU).filter(ADU.type == 'claim').count()
        cache = _load_cache(CLAIMS_CACHE_FILE)
        if cache and cache.get('total') == total:
            return cache['test']
        training = _get_claims(session, split='training')
        test_ = _get_claims(session, split='test')
        benchmark = _get_claims(session, split='benchmark')
        _save_cache(CLAIMS_CACHE_FILE, {'total': total, 'training': training, 'test': test_, 'benchmark': benchmark})
        return test_

def get_benchmark_claims() -> list[ADU]:
    """Return the last 10% of claims, using cache if enabled and up-to-date."""
    with get_session() as session:
        if not CACHE_ENABLED:
            return _get_claims(session, split='benchmark')
        total = session.query(ADU).filter(ADU.type == 'claim').count()
        cache = _load_cache(CLAIMS_CACHE_FILE)
        if cache and cache.get('total') == total and 'benchmark' in cache:
            return cache['benchmark']
        training = _get_claims(session, split='training')
        test_ = _get_claims(session, split='test')
        benchmark = _get_claims(session, split='benchmark')
        _save_cache(CLAIMS_CACHE_FILE, {'total': total, 'training': training, 'test': test_, 'benchmark': benchmark})
        return benchmark
    
def get_training_data() -> tuple[list[ADU], list[ADU], list[str]]:
    """Return (claims, premises, relationship_category (stance_pro/stance_con) ) for training split, using cache if enabled and fresh."""

    with get_session() as session:
        if not CACHE_ENABLED:
            return _get_data(session, training=True)
        db_total_claims = session.query(ADU).filter(ADU.type == 'claim').count()
        cache = _load_cache(DATA_CACHE_FILE)
        if cache and cache.get('db_total_claims_when_cached') == db_total_claims: # More robust cache check
            return cache['training']
        
        train_ = _get_data(session, split='training')
        test_ = _get_data(session, split='test')
        benchmark_ = _get_data(session, split='benchmark')
        _save_cache(DATA_CACHE_FILE, {'db_total_claims_when_cached': db_total_claims, 'training': train_, 'test': test_, 'benchmark':benchmark_})
        return train_


def get_test_data() -> tuple[list[ADU], list[ADU], list[str]]:
    """Return (claims, premises, relationship_category (stance_pro/stance_con) ) for test split, using cache if enabled and fresh."""

    with get_session() as session:
        if not CACHE_ENABLED:
            return _get_data(session, training=False)
        db_total_claims = session.query(ADU).filter(ADU.type == 'claim').count()
        cache = _load_cache(DATA_CACHE_FILE)
        if cache and cache.get('db_total_claims_when_cached') == db_total_claims: # More robust cache check
            return cache['test']

        # If cache is invalid or test data not present, might need to re-fetch both if cache structure allows
        train_ = _get_data(session, split='training') # Or fetch only test if cache structure allows
        test_ = _get_data(session, split='test')
        benchmark_ = _get_data(session, split='benchmark')
        _save_cache(DATA_CACHE_FILE, {'db_total_claims_when_cached': db_total_claims, 'training': train_, 'test': test_, 'benchmark':benchmark_})
        return test_

def get_benchmark_data() -> tuple[list[ADU], list[ADU], list[str]]:
    """Return (claims, premises, relationship_category (stance_pro/stance_con) ) for benchmark split, using cache if enabled and fresh."""
    with get_session() as session:
        if not CACHE_ENABLED:
            return _get_data(session, split='benchmark')
        total = session.query(ADU).filter(ADU.type == 'claim').count()
        cache = _load_cache(DATA_CACHE_FILE)
        if cache and cache.get('db_total_claims_when_cached') == total and 'benchmark' in cache:
            return cache['benchmark']
        train_ = _get_data(session, split='training')
        test_ = _get_data(session, split='test')
        benchmark_ = _get_data(session, split='benchmark')
        _save_cache(DATA_CACHE_FILE, {'db_total_claims_when_cached': total, 'training': train_, 'test': test_, 'benchmark': benchmark_})
        return benchmark_

def get_sharded_training_data(max_per_shard: int, num_shards: int) -> list[tuple[list[ADU], list[ADU], list[str]]]:
    """
    Return a list of `num_shards` tuples (claims, premises, categories), each containing
    exactly `max_per_shard` items, drawn consecutively from the training split.

    Raises:
        ValueError: if max_per_shard <= 0, num_shards <= 0,
                    or nicht genug Daten, um alle Shards komplett zu füllen.
    """
    if max_per_shard <= 0 or num_shards <= 0:
        raise ValueError("max_per_shard und num_shards müssen positive Ganzzahlen sein.")
    
    # Holt alle Trainingsdaten
    claims, premises, categories = get_training_data()
    total = len(claims)
    required = max_per_shard * num_shards
    
    if total < required:
        raise ValueError(
            f"Nicht genug Trainingsdaten: benötigt {required}, vorhanden {total}."
        )
    
    shards: list[tuple[list[ADU], list[ADU], list[str]]] = []
    for i in range(num_shards):
        start = i * max_per_shard
        end = start + max_per_shard
        # Die Slices haben hier garantiert die Länge max_per_shard
        shards.append((
            claims[start:end],
            premises[start:end],
            categories[start:end]
        ))
    
    return shards

def get_benchmark_data_details(number_of_premises: int, specific_ids: list[int] = None) -> tuple[list[ADU], list[list[ADU]], list[list[str]]]:
    """
    Return (claims, premises_list, categories_list) for the benchmark split (last 10% of claims),
    with up to `number_of_premises` premises per claim. If `specific_ids` is provided, only claims
    with those IDs are returned. If a claim has fewer premises than `number_of_premises`, all available
    premises are returned.
    """
    with get_session() as session:
        claims = _get_claims(session, split='benchmark')
        if specific_ids:
            claims = [c for c in claims if c.id in specific_ids]
        claim_ids = [c.id for c in claims]
        rows = (
            session
            .query(ADU, Relationship.category, Relationship.to_adu_id)
            .join(Relationship, Relationship.from_adu_id == ADU.id)
            .filter(ADU.type == 'premise', Relationship.to_adu_id.in_(claim_ids))
            .all()
        )
        premises, categories, to_ids = zip(*rows) if rows else ([], [], [])
        grouped = defaultdict(list)
        for premise, cat, cid in zip(premises, categories, to_ids):
            grouped[cid].append((premise, cat))

        output_claims = []
        output_premises = []
        output_categories = []  
        for claim in claims:
            output_claims.append(claim)
            items = grouped.get(claim.id, [])
            if number_of_premises <= 0:
                limited = []
            else:
                limited = items[:number_of_premises]
            ps = [p for p, _ in limited]
            cs = [cat for _, cat in limited]
            output_premises.append(ps)
            output_categories.append(cs)

        return output_claims, output_premises, output_categories


def get_quality_data(claims_premises_dict: dict[int, list[int]]=data) -> tuple[list[ADU], list[list[ADU]], list[list[str]]]:
    """
    Return (claims, premises_list, categories_list) for given claim-premise mappings.
    The input is a dictionary where the key is a claim_id and the value is a list of premise_ids.
    """
    with get_session() as session:
        claim_ids = list(claims_premises_dict.keys())
        premise_ids = list({pid for pids in claims_premises_dict.values() for pid in pids})
        claims = session.query(ADU).filter(ADU.id.in_(claim_ids)).all()
        claims_by_id = {c.id: c for c in claims}
        premises = session.query(ADU).filter(ADU.id.in_(premise_ids)).all()
        premises_by_id = {p.id: p for p in premises}
        rows = (
            session
            .query(Relationship.from_adu_id, Relationship.to_adu_id, Relationship.category)
            .filter(
                Relationship.to_adu_id.in_(claim_ids),
                Relationship.from_adu_id.in_(premise_ids)
            )
            .all()
        )
        category_lookup = {(from_id, to_id): cat for from_id, to_id, cat in rows}
        output_claims = []
        output_premises = []
        output_categories = []

        for claim_id, premise_list in claims_premises_dict.items():
            claim = claims_by_id.get(claim_id)
            if not claim:
                continue 

            current_premises = []
            current_categories = []

            for pid in premise_list:
                premise = premises_by_id.get(pid)
                if not premise:
                    continue  # Skip if premise is missing

                current_premises.append(premise)
                category = category_lookup.get((pid, claim_id), None)
                current_categories.append(category)

            output_claims.append(claim)
            output_premises.append(current_premises)
            output_categories.append(current_categories)

        return output_claims, output_premises, output_categories


def get_benchmark_data_for_evaluation(max_samples: int = 100) -> list[dict]:
    """
    Wrapper function that transforms the database output format into the format expected by the benchmark.
    Returns a list of dictionaries with the structure expected by the benchmark.
    
    Args:
        max_samples: Maximum number of samples to return (default: 100)
    """
    try:
        # Try to get data from database
        claims, premises, stances = get_benchmark_data()
        
        # Limit the number of samples
        if max_samples and max_samples > 0:
            claims = claims[:max_samples]
            premises = premises[:max_samples]
            stances = stances[:max_samples]
        
        # Transform the data into the expected format
        benchmark_data = []
        
        for i, (claim, premise_list, stance) in enumerate(zip(claims, premises, stances)):
            # Create the text by combining claim and premises
            text_parts = [claim.text]
            for premise in premise_list:
                text_parts.append(premise.text)
            text = " ".join(text_parts)
            
            # Create ground truth structure
            ground_truth = {
                'adus': [
                    {'text': claim.text, 'type': 'claim', 'id': claim.id}
                ] + [
                    {'text': premise.text, 'type': 'premise', 'id': premise.id}
                    for premise in premise_list
                ],
                'stance': stance,
                'relationships': [
                    {'claim_id': claim.id, 'premise_ids': [p.id for p in premise_list]}
                ] if premise_list else []
            }
            
            # Create metadata
            metadata = {
                'source': 'database',
                'domain': 'argument_mining',
                'claim_id': claim.id,
                'premise_count': len(premise_list)
            }
            
            benchmark_data.append({
                'id': i + 1,
                'text': text,
                'ground_truth': ground_truth,
                'metadata': metadata
            })
        
        return benchmark_data
        
    except Exception as e:
        # Fallback to sample data if database is not available
        print(f"Warning: Database not available, using sample data. Error: {e}")
        return get_sample_benchmark_data(max_samples)


def get_sample_benchmark_data(max_samples: int = 100) -> list[dict]:
    """
    Returns sample benchmark data for testing when database is not available.
    
    Args:
        max_samples: Maximum number of samples to return (default: 100)
    """
    # Generate more sample data to meet the max_samples requirement
    sample_templates = [
        {
            'text': 'Climate change is real and caused by human activities. Scientific evidence shows increasing temperatures. Carbon emissions from fossil fuels contribute to global warming.',
            'ground_truth': {
                'adus': [
                    {'text': 'Climate change is real and caused by human activities', 'type': 'claim', 'id': 1},
                    {'text': 'Scientific evidence shows increasing temperatures', 'type': 'premise', 'id': 2},
                    {'text': 'Carbon emissions from fossil fuels contribute to global warming', 'type': 'premise', 'id': 3}
                ],
                'stance': 'pro',
                'relationships': [{'claim_id': 1, 'premise_ids': [2, 3]}]
            },
            'metadata': {
                'source': 'sample',
                'domain': 'climate_change',
                'claim_id': 1,
                'premise_count': 2
            }
        },
        {
            'text': 'Social media has negative effects on mental health. Studies show increased anxiety and depression. Screen time reduces face-to-face interactions.',
            'ground_truth': {
                'adus': [
                    {'text': 'Social media has negative effects on mental health', 'type': 'claim', 'id': 4},
                    {'text': 'Studies show increased anxiety and depression', 'type': 'premise', 'id': 5},
                    {'text': 'Screen time reduces face-to-face interactions', 'type': 'premise', 'id': 6}
                ],
                'stance': 'con',
                'relationships': [{'claim_id': 4, 'premise_ids': [5, 6]}]
            },
            'metadata': {
                'source': 'sample',
                'domain': 'mental_health',
                'claim_id': 4,
                'premise_count': 2
            }
        },
        {
            'text': 'Remote work improves productivity and work-life balance. Employees report higher job satisfaction. Commute time is eliminated, reducing stress.',
            'ground_truth': {
                'adus': [
                    {'text': 'Remote work improves productivity and work-life balance', 'type': 'claim', 'id': 7},
                    {'text': 'Employees report higher job satisfaction', 'type': 'premise', 'id': 8},
                    {'text': 'Commute time is eliminated, reducing stress', 'type': 'premise', 'id': 9}
                ],
                'stance': 'pro',
                'relationships': [{'claim_id': 7, 'premise_ids': [8, 9]}]
            },
            'metadata': {
                'source': 'sample',
                'domain': 'workplace',
                'claim_id': 7,
                'premise_count': 2
            }
        },
        {
            'text': 'Artificial intelligence will replace many jobs. Automation reduces employment opportunities. Human workers become obsolete in many industries.',
            'ground_truth': {
                'adus': [
                    {'text': 'Artificial intelligence will replace many jobs', 'type': 'claim', 'id': 10},
                    {'text': 'Automation reduces employment opportunities', 'type': 'premise', 'id': 11},
                    {'text': 'Human workers become obsolete in many industries', 'type': 'premise', 'id': 12}
                ],
                'stance': 'con',
                'relationships': [{'claim_id': 10, 'premise_ids': [11, 12]}]
            },
            'metadata': {
                'source': 'sample',
                'domain': 'technology',
                'claim_id': 10,
                'premise_count': 2
            }
        },
        {
            'text': 'Electric vehicles are better for the environment. They produce zero emissions during operation. Battery production has environmental costs.',
            'ground_truth': {
                'adus': [
                    {'text': 'Electric vehicles are better for the environment', 'type': 'claim', 'id': 13},
                    {'text': 'They produce zero emissions during operation', 'type': 'premise', 'id': 14},
                    {'text': 'Battery production has environmental costs', 'type': 'premise', 'id': 15}
                ],
                'stance': 'pro',
                'relationships': [{'claim_id': 13, 'premise_ids': [14, 15]}]
            },
            'metadata': {
                'source': 'sample',
                'domain': 'environment',
                'claim_id': 13,
                'premise_count': 2
            }
        }
    ]
    
    # Generate the requested number of samples by cycling through templates
    benchmark_data = []
    for i in range(min(max_samples, len(sample_templates) * 20)):  # Cap at 20x templates to avoid infinite loop
        template = sample_templates[i % len(sample_templates)]
        
        # Create a copy with updated IDs
        sample = {
            'id': i + 1,
            'text': template['text'],
            'ground_truth': {
                'adus': [
                    {**adu, 'id': adu['id'] + (i * 10)}  # Offset IDs to make them unique
                    for adu in template['ground_truth']['adus']
                ],
                'stance': template['ground_truth']['stance'],
                'relationships': [
                    {
                        'claim_id': rel['claim_id'] + (i * 10),
                        'premise_ids': [pid + (i * 10) for pid in rel['premise_ids']]
                    }
                    for rel in template['ground_truth']['relationships']
                ]
            },
            'metadata': {
                **template['metadata'],
                'claim_id': template['metadata']['claim_id'] + (i * 10)
            }
        }
        
        benchmark_data.append(sample)
        
        if len(benchmark_data) >= max_samples:
            break
    
    return benchmark_data
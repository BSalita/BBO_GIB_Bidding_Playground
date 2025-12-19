"""
Bidding Models Interface for the Bidding Arena.

This module provides an abstract base class for bidding models, making it easy
for the bridge community to add new models (neural networks, random forests,
fuzzy logic, etc.) and compare them in the Bidding Arena.

To add a new model:
1. Create a new class that inherits from BiddingModel
2. Implement the required methods (just get_auction!)
3. Register your model using MODEL_REGISTRY.register()

Example:
    class MyNeuralNetModel(BiddingModel):
        @property
        def name(self) -> str:
            return "NN"
        
        @property
        def display_name(self) -> str:
            return "Neural Network"
        
        @property
        def description(self) -> str:
            return "LSTM-based bidding model trained on 1M deals"
        
        def get_auction(self, deal, dealer, state) -> str | None:
            # Your inference code here
            return self.model.predict(deal)
    
    # Register the model
    MODEL_REGISTRY.register(MyNeuralNetModel())
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import re


class BiddingModel(ABC):
    """Abstract base class for all bidding models.
    
    Implement this interface to add a new bidding model to the Bidding Arena.
    
    CONTESTANTS ONLY NEED TO IMPLEMENT:
    - name, display_name, description (metadata)
    - get_auction() - the auction your model produces
    
    DD SCORES AND EV ARE AUTOMATICALLY CALCULATED by the arena infrastructure
    based on the auction/contract your model produces. This ensures fair,
    consistent evaluation across all models.
    """
    
    # =========================================================================
    # REQUIRED: Model Metadata
    # =========================================================================
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this model.
        
        Used in API requests and column naming (e.g., 'Rules', 'NN', 'RF').
        Should be short and alphanumeric.
        """
        pass
    
    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for display in UI.
        
        Example: 'Rule-Based GIB', 'Neural Network v2', 'Random Forest'
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Detailed description of the model.
        
        Include training data, architecture, or methodology as appropriate.
        """
        pass
    
    @property
    def version(self) -> str:
        """Version string for this model (optional, defaults to '1.0')."""
        return "1.0"
    
    @property
    def author(self) -> str:
        """Author or organization that created this model (optional)."""
        return "Unknown"
    
    # =========================================================================
    # REQUIRED: Core Bidding Method
    # =========================================================================
    
    @abstractmethod
    def get_auction(
        self,
        deal: Dict[str, Any],
        dealer: str,
        state: Dict[str, Any],
    ) -> Optional[str]:
        """Get the auction string this model would produce for a deal.
        
        THIS IS THE MAIN METHOD CONTESTANTS MUST IMPLEMENT.
        
        Args:
            deal: Deal data including:
                - Hand_N, Hand_E, Hand_S, Hand_W: PBN format hands (e.g., "AKQ.JT9.876.5432")
                - Vul: Vulnerability ("None", "NS", "EW", "Both")
                - Other deal metadata as needed
            dealer: The dealer direction ('N', 'E', 'S', 'W').
            state: Shared API state (for accessing lookup tables if needed).
        
        Returns:
            Auction string (e.g., '1N-p-3N-p-p-p') or None if unavailable.
            Use lowercase 'p' for pass, 'x' for double, 'xx' for redouble.
        """
        pass
    
    # =========================================================================
    # OPTIONAL: Override if needed
    # =========================================================================
    
    def get_contract(
        self,
        deal: Dict[str, Any],
        dealer: str,
        state: Dict[str, Any],
    ) -> Optional[str]:
        """Get the final contract this model would reach.
        
        DEFAULT: Automatically derived from get_auction() using standard rules.
        Override only if you need custom contract extraction logic.
        
        Returns:
            Contract string (e.g., '3NT', '4S', '6Hx') or None if passed out.
        """
        from bbo_bidding_queries_lib import get_ai_contract
        auction = self.get_auction(deal, dealer, state)
        if auction:
            return get_ai_contract(auction, dealer)
        return None
    
    def is_available(self, state: Dict[str, Any]) -> bool:
        """Check if this model is available for use.
        
        Override to check for required data files, model weights, etc.
        
        Args:
            state: Shared API state.
        
        Returns:
            True if the model can be used, False otherwise.
        """
        return True
    
    def get_column_prefix(self) -> str:
        """Get the column name prefix for this model.
        
        Used for naming output columns like '{prefix}_Bid', '{prefix}_Contract'.
        Defaults to the model name.
        """
        return self.name
    
    # =========================================================================
    # ARENA-CALCULATED: Do NOT override these in contestant models
    # =========================================================================
    
    def get_dd_score(
        self,
        deal: Dict[str, Any],
        dealer: str,
        state: Dict[str, Any],
    ) -> Optional[int]:
        """Get the double-dummy score for this model's contract.
        
        AUTOMATICALLY CALCULATED by the arena from the deal's DD table
        based on the contract produced by get_contract().
        
        Contestants should NOT override this method.
        """
        from bbo_bidding_queries_lib import get_dd_score_for_auction
        auction = self.get_auction(deal, dealer, state)
        if auction:
            return get_dd_score_for_auction(auction, dealer, deal)
        return None
    
    def get_ev(
        self,
        deal: Dict[str, Any],
        dealer: str,
        state: Dict[str, Any],
    ) -> Optional[float]:
        """Get the expected value for this model's contract.
        
        AUTOMATICALLY CALCULATED by the arena from the deal's EV table
        based on the contract produced by get_contract().
        
        Contestants should NOT override this method.
        """
        from bbo_bidding_queries_lib import get_ev_for_auction
        auction = self.get_auction(deal, dealer, state)
        if auction:
            return get_ev_for_auction(auction, dealer, deal)
        return None


class ModelRegistry:
    """Registry for bidding models.
    
    Models register themselves here to be available in the Bidding Arena.
    """
    
    def __init__(self):
        self._models: Dict[str, BiddingModel] = {}
    
    def register(self, model: BiddingModel) -> None:
        """Register a bidding model.
        
        Args:
            model: The model instance to register.
        
        Raises:
            ValueError: If a model with the same name is already registered.
        """
        name = model.name
        if name in self._models:
            raise ValueError(f"Model '{name}' is already registered")
        self._models[name] = model
        print(f"[models] Registered bidding model: {model.display_name} ({name})")
    
    def unregister(self, name: str) -> None:
        """Unregister a bidding model.
        
        Args:
            name: The model name to unregister.
        """
        if name in self._models:
            del self._models[name]
    
    def get(self, name: str) -> Optional[BiddingModel]:
        """Get a registered model by name.
        
        Args:
            name: The model name.
        
        Returns:
            The model instance or None if not found.
        """
        return self._models.get(name)
    
    def get_all(self) -> Dict[str, BiddingModel]:
        """Get all registered models.
        
        Returns:
            Dict mapping model names to model instances.
        """
        return dict(self._models)
    
    def list_models(self) -> List[Dict[str, str]]:
        """List all registered models with their metadata.
        
        Returns:
            List of dicts with name, display_name, description, version, author.
        """
        return [
            {
                "name": m.name,
                "display_name": m.display_name,
                "description": m.description,
                "version": m.version,
                "author": m.author,
            }
            for m in self._models.values()
        ]
    
    def is_valid_model(self, name: str) -> bool:
        """Check if a model name is valid (registered).
        
        Args:
            name: The model name to check.
        
        Returns:
            True if the model is registered.
        """
        return name in self._models


# Global model registry
MODEL_REGISTRY = ModelRegistry()


# ---------------------------------------------------------------------------
# Built-in Models
# ---------------------------------------------------------------------------


class RulesModel(BiddingModel):
    """Rule-based bidding model using GIB-style auction tables.
    
    This model looks up auctions from the pre-computed bt_seat1_df table,
    which contains rule-based bidding sequences generated by GIB.
    """
    
    @property
    def name(self) -> str:
        return "Rules"
    
    @property
    def display_name(self) -> str:
        return "Rule-Based (GIB)"
    
    @property
    def description(self) -> str:
        return "GIB-style rule-based bidding using pre-computed auction tables"
    
    @property
    def author(self) -> str:
        return "GIB/BBO"
    
    def get_auction(
        self,
        deal: Dict[str, Any],
        dealer: str,
        state: Dict[str, Any],
    ) -> Optional[str]:
        """Look up the auction from bt_seat1_df."""
        import polars as pl
        
        bt_seat1_df = state.get("bt_seat1_df")
        if bt_seat1_df is None:
            return None
        
        # Get the deal's actual bid to find matching auction
        bid = deal.get("bid")
        if isinstance(bid, list):
            bid_str = "-".join(str(b) for b in bid)
        else:
            bid_str = str(bid) if bid else ""
        
        # Strip leading passes for lookup
        auction_for_search = re.sub(r"^(p-)+", "", bid_str.lower()) if bid_str else ""
        
        # Look up in bt_seat1_df
        bt_lookup_df = bt_seat1_df
        if "is_completed_auction" in bt_lookup_df.columns:
            bt_lookup_df = bt_lookup_df.filter(pl.col("is_completed_auction"))
        
        bt_match = bt_lookup_df.filter(
            pl.col("Auction").cast(pl.Utf8).str.to_lowercase() == auction_for_search
        )
        if bt_match.height == 0 and not auction_for_search.endswith("-p-p-p"):
            bt_match = bt_lookup_df.filter(
                pl.col("Auction").cast(pl.Utf8).str.to_lowercase() == auction_for_search + "-p-p-p"
            )
        
        if bt_match.height > 0:
            return bt_match.row(0, named=True).get("Auction")
        return None
    
    def is_available(self, state: Dict[str, Any]) -> bool:
        """Check if bt_seat1_df is loaded."""
        return state.get("bt_seat1_df") is not None


class ActualModel(BiddingModel):
    """Model representing actual bids made in the dataset.
    
    This is not a predictive model - it simply returns the actual
    auction that occurred in each deal. Used as a baseline for comparison.
    
    Note: This model overrides get_dd_score and get_ev to use the
    precomputed values for the actual contract, since the deal already
    has these values stored.
    """
    
    @property
    def name(self) -> str:
        return "Actual"
    
    @property
    def display_name(self) -> str:
        return "Actual (Human)"
    
    @property
    def description(self) -> str:
        return "Actual bids made by human players in the BBO dataset"
    
    @property
    def author(self) -> str:
        return "BBO Players"
    
    def get_auction(
        self,
        deal: Dict[str, Any],
        dealer: str,
        state: Dict[str, Any],
    ) -> Optional[str]:
        """Return the actual bid from the deal."""
        bid = deal.get("bid")
        if isinstance(bid, list):
            return "-".join(str(b) for b in bid)
        return str(bid) if bid else None
    
    def get_contract(
        self,
        deal: Dict[str, Any],
        dealer: str,
        state: Dict[str, Any],
    ) -> Optional[str]:
        """Return the actual contract from the deal (precomputed)."""
        return deal.get("Contract")
    
    def get_dd_score(
        self,
        deal: Dict[str, Any],
        dealer: str,
        state: Dict[str, Any],
    ) -> Optional[int]:
        """Return the precomputed DD score for the actual contract."""
        # Use precomputed value since it's already in the deal
        return deal.get("DD_Score_Declarer")
    
    def get_ev(
        self,
        deal: Dict[str, Any],
        dealer: str,
        state: Dict[str, Any],
    ) -> Optional[float]:
        """Return the precomputed EV for the actual contract."""
        # Use precomputed value since it's already in the deal
        return deal.get("EV_Score_Declarer")
    
    def is_available(self, state: Dict[str, Any]) -> bool:
        """Always available if deals are loaded."""
        return state.get("deal_df") is not None


# ---------------------------------------------------------------------------
# Register Built-in Models
# ---------------------------------------------------------------------------

# Register the built-in models
MODEL_REGISTRY.register(RulesModel())
MODEL_REGISTRY.register(ActualModel())


# ---------------------------------------------------------------------------
# Template for New Models
# ---------------------------------------------------------------------------

"""
# ==========================================================================
# TEMPLATE: How to Add a New Bidding Model to the Bidding Arena
# ==========================================================================

CONTESTANTS ONLY NEED TO IMPLEMENT:
  1. name, display_name, description (metadata)
  2. get_auction() - return the auction your model produces

DD SCORES AND EV ARE AUTOMATICALLY CALCULATED by the arena!
You do NOT need to implement get_dd_score() or get_ev().

# ==========================================================================
# Example: Neural Network Model
# ==========================================================================

# 1. Create a new file (e.g., my_nn_model.py)

from bidding_models import BiddingModel, MODEL_REGISTRY

class MyNeuralNetModel(BiddingModel):
    '''My custom neural network bidding model.'''
    
    def __init__(self, model_path: str = "models/nn_bidder.pt"):
        self.model_path = model_path
        self._model = None  # Lazy load
    
    # ---- REQUIRED: Metadata ----
    
    @property
    def name(self) -> str:
        return "NN"  # Short unique identifier
    
    @property
    def display_name(self) -> str:
        return "Neural Network v1"
    
    @property
    def description(self) -> str:
        return "LSTM-based bidding model trained on 1M BBO deals"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def author(self) -> str:
        return "Your Name"
    
    # ---- REQUIRED: Core bidding method ----
    
    def get_auction(self, deal, dealer, state):
        '''
        THE ONLY METHOD YOU NEED TO IMPLEMENT!
        
        Args:
            deal: Dict with Hand_N, Hand_E, Hand_S, Hand_W (PBN format),
                  Vul ("None", "NS", "EW", "Both"), etc.
            dealer: "N", "E", "S", or "W"
            state: API state (for accessing lookup tables if needed)
        
        Returns:
            Auction string like "1N-p-3N-p-p-p"
            Use lowercase: p=pass, x=double, xx=redouble
        '''
        model = self._load_model()
        
        # Convert deal to model input format
        hands = [deal.get(f"Hand_{d}") for d in ["N", "E", "S", "W"]]
        vul = deal.get("Vul", "None")
        
        # Run inference
        with torch.no_grad():
            bids = model.predict(hands, dealer, vul)
        
        return "-".join(bids)
    
    # ---- OPTIONAL: Override if needed ----
    
    def is_available(self, state):
        '''Check if model weights are available.'''
        import os
        return os.path.exists(self.model_path)
    
    # ---- PRIVATE: Helper methods ----
    
    def _load_model(self):
        '''Lazy load the model weights.'''
        if self._model is None:
            import torch
            self._model = torch.load(self.model_path)
            self._model.eval()
        return self._model


# 2. Register your model (do this at module import time or in app startup)
# MODEL_REGISTRY.register(MyNeuralNetModel())

# 3. Your model is now available in the Bidding Arena!
#    GET  /bidding-models                    -> See your model listed
#    POST /bidding-arena {"model_a": "NN", "model_b": "Rules"}

# ==========================================================================
# Deal Data Available to Your Model
# ==========================================================================
#
# deal dict contains:
#   - Hand_N, Hand_E, Hand_S, Hand_W: PBN format (e.g., "AKQ.JT9.876.5432")
#   - Vul: Vulnerability ("None", "NS", "EW", "Both")
#   - Dealer: Dealer direction ("N", "E", "S", "W")
#   - HCP_N, HCP_E, HCP_S, HCP_W: High card points
#   - Plus many other precomputed features...
#
# state dict contains:
#   - deal_df: Full deals DataFrame
#   - bt_seat1_df: Bidding table DataFrame
#   - Other API state as needed
#
# ==========================================================================
"""


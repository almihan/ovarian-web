"""Entity-extraction integration point.

Use ``request.app.state.cell_model`` for CellExLink so the model is not loaded
again for each paper or request. Gene and chemical model bundles can be added to
``app.state`` using the same startup pattern.
"""

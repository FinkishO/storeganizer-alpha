"""
3D Visualization integration for Storeganizer (PLACEHOLDER).

This module is prepared for integrating Storeganizer's 3D visualization platform.

Integration options:
1. iframe embed of Dimitri's visualization URLs
2. API integration with Storeganizer visualization service
3. Custom 3D renderer (future consideration)

Current status: ALPHA - 2D visualization only
Next step: Workshop with Dimitri to understand URL parameter structure
"""

from typing import Dict, List, Optional
from config import storeganizer as config


def generate_visualization_url(
    size_category: str,
    planogram_data: Optional[Dict] = None,
) -> str:
    """
    Generate URL for Storeganizer 3D visualization.

    Args:
        size_category: "large", "medium", "small", or "xs"
        planogram_data: Optional dict with bay/column/SKU data

    Returns:
        URL string for iframe embedding

    Note:
        This is a placeholder. URL parameter structure needs to be
        determined with Dimitri during Belgium workshop.
    """
    if size_category not in config.VISUALIZATION_URLS:
        size_category = "medium"  # Default fallback

    base_url = config.VISUALIZATION_URLS[size_category]

    # TODO: After workshop with Dimitri, append query parameters
    # based on planogram_data to show actual SKU allocation
    # Example (hypothetical):
    # params = []
    # if planogram_data:
    #     params.append(f"bays={planogram_data['bay_count']}")
    #     params.append(f"config={planogram_data['configuration_id']}")
    #     # etc.
    # if params:
    #     base_url += "?" + "&".join(params)

    return base_url


def embed_3d_viewer(
    planogram_data: Dict,
    width: str = "100%",
    height: str = "600px",
) -> str:
    """
    Generate HTML iframe code for 3D visualization.

    Args:
        planogram_data: Dict with planogram details
        width: iframe width (CSS)
        height: iframe height (CSS)

    Returns:
        HTML string for iframe embed

    Usage in Streamlit:
        st.components.v1.html(embed_3d_viewer(data), height=650)
    """
    if not config.ENABLE_3D_VIEWER:
        return (
            "<div style='padding: 40px; text-align: center; background: #f0f0f0; "
            "border-radius: 10px; color: #666;'>"
            "<h3>3D Visualization</h3>"
            "<p>Coming soon - will integrate with Storeganizer visualization platform</p>"
            "<p>Links provided by Dimitri:</p>"
            f"<ul style='list-style: none;'>"
            f"<li><a href='{config.VISUALIZATION_URLS['large']}' target='_blank'>Large configuration</a></li>"
            f"<li><a href='{config.VISUALIZATION_URLS['medium']}' target='_blank'>Medium configuration</a></li>"
            f"<li><a href='{config.VISUALIZATION_URLS['small']}' target='_blank'>Small configuration</a></li>"
            f"</ul>"
            "</div>"
        )

    # When enabled, generate actual iframe
    size_category = planogram_data.get("size_category", "medium")
    url = generate_visualization_url(size_category, planogram_data)

    iframe_html = f"""
    <iframe
        src="{url}"
        width="{width}"
        height="{height}"
        style="border: none; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"
        allow="fullscreen"
    ></iframe>
    """

    return iframe_html


def get_configuration_suggestions(bay_count: int, sku_count: int) -> Dict[str, str]:
    """
    Suggest Storeganizer configuration size based on requirements.

    Args:
        bay_count: Number of bays in planogram
        sku_count: Number of SKUs

    Returns:
        Dict with size_category and reasoning

    Note:
        Thresholds are placeholder - refine with Dimitri's input
    """
    if bay_count >= 10 or sku_count >= 400:
        return {
            "size_category": "large",
            "reason": "High capacity requirements - large configuration recommended",
            "visualization_url": config.VISUALIZATION_URLS["large"],
        }
    elif bay_count >= 6 or sku_count >= 200:
        return {
            "size_category": "medium",
            "reason": "Medium capacity - standard configuration",
            "visualization_url": config.VISUALIZATION_URLS["medium"],
        }
    elif bay_count >= 3 or sku_count >= 100:
        return {
            "size_category": "small",
            "reason": "Smaller operation - compact configuration",
            "visualization_url": config.VISUALIZATION_URLS["small"],
        }
    else:
        return {
            "size_category": "xs",
            "reason": "Minimal requirements - extra-small configuration",
            "visualization_url": config.VISUALIZATION_URLS["xs"],
        }


# ===========================
# FUTURE IMPLEMENTATION NOTES
# ===========================

"""
Questions for Dimitri during Belgium workshop:

1. URL Parameter Structure:
   - How do we pass bay count, configuration type, SKU details?
   - Can we pre-populate specific pocket assignments?
   - Is there an API endpoint, or just static visualization URLs?

2. Customization Integration:
   - How does their customization configurator work?
   - Can we link directly from planogram to customization flow?
   - Do they have an API for retrieving product specs?

3. Internal UI:
   - What parts of their internal UI need modernization?
   - Should this tool integrate with their existing systems?
   - Data exchange formats (API, file export, database)?

4. Commercial Model:
   - Do they want this as standalone tool or integrated into their platform?
   - White-label vs. co-branded?
   - Hosting preferences (their servers, Fred's server, cloud)?
"""

HIDE_STREAMLIT_STYLE = """
<style>

/* Hide Main Menu */
#MainMenu {
    visibility: hidden;
}

/* Hide Header */
header {
    visibility: hidden;
}

/* Hide Footer */
footer {
    visibility: hidden;
}

/* Hide Top Toolbar */
[data-testid="stToolbar"] {
    display: none;
}

/* Hide Running Status */
[data-testid="stStatusWidget"] {
    display: none;
}

/* Hide Top Decoration */
[data-testid="stDecoration"] {
    display: none;
}

</style>
"""
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc

from callbacks import create_callbacks


def create_layout() -> html.Div:
    nav_bar = dbc.NavbarSimple(
        brand="Construction Dataset Visualiser",
        brand_href="#",
        class_name="sticky-top",
    )

    select_dataset = html.Div(
        [
            html.Br(),
            html.H6("Select the dataset:"),
            dbc.Select(
                id="select",
                value="luo2020",
                options=[
                    {
                        "label": "Roberts and Golparvar-Fard (2019) - End-to-end vision-based detection, tracking and "
                        "activity analysis of earthmoving equipment filmed at ground level",
                        "value": "roberts2019",
                    },
                    {
                        "label": "Luo et al. (2020) - Full body pose estimation of construction equipment using computer "
                        "vision and deep learning techniques",
                        "value": "luo2020",
                    },
                    {
                        "label": "Bang and Kim (2020) - Context-based information generation for managing UAV-acquired data "
                        "using image captioning",
                        "value": "bang2020",
                    },
                    {"label": "Xiao and Kang (2021) - Alberta Construction Image Dataset (ACID)", "value": "ACID"},
                    {"label": "An et al. (2021) - Moving Objects in the Construction Site (MOCS)", "value": "MOCS"},
                ],
            ),
        ]
    )

    graph = html.Div(
        [
            dcc.Graph(id="graph", style={"height": "60vh"}),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Slider(
                            id="slider-page",
                            min=1,
                            max=1000,
                            value=1,
                            step=1,
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                        )
                    ),
                    dbc.Col(dbc.Input(id="input-page", type="number", min=1, max=1000, value=1), width=2),
                ],
                align="center",
                justify="center",
            ),
        ]
    )

    return html.Div(
        [
            nav_bar,
            dbc.Container(
                [
                    select_dataset,
                    graph,
                ],
                class_name="mb-5",
            ),
        ]
    )


def create_app() -> Dash:
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
    app.layout = create_layout()
    create_callbacks(app)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run()

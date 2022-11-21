import dash
import numpy as np
import plotly.graph_objects as go
from dash import Output, Input
from torch.utils import data
from torchvision.datasets import VisionDataset

from datasets import Luo2020, Bang2020, ACID, MOCS


def create_callbacks(app: dash.Dash) -> None:
    datasets = {
        "luo2020": Luo2020("data/"),
        "bang2020": Bang2020("data/"),
        "ACID": ACID("data/"),
        "MOCS": data.ConcatDataset(
            [
                MOCS("data/", "train"),
                MOCS("data/", "val"),
            ]
        ),
    }

    @app.callback(
        [
            Output("graph", "figure"),
            Output("slider-page", "value"),
            Output("slider-page", "max"),
            Output("input-page", "value"),
            Output("input-page", "max"),
        ],
        [Input("select", "value"), Input("slider-page", "value"), Input("input-page", "value")],
    )
    def update_graph(dataset_name: str, slider_value, input_value):
        # TODO: move all this to classes

        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
        )
        if dash.ctx.triggered_id == "slider-page":
            page = slider_value
        elif dash.ctx.triggered_id == "input-page":
            page = input_value
        else:
            page = 1  # dataset changed, reset to first page

        if dataset_name not in datasets.keys():
            page = 0
            max_value = 0
            return fig, page, max_value, page, max_value
        else:
            dataset: VisionDataset = datasets[dataset_name]
            max_value = len(dataset)
            idx = page - 1
            img, target = dataset[idx]

            fig.update_xaxes(
                range=[0, img.width],
                showgrid=False,
            )
            fig.update_yaxes(range=[0, img.height], scaleanchor="x", scaleratio=1, showgrid=False, autorange="reversed")

            fig.add_layout_image(
                source=img,
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=img.width,
                sizey=img.height,
                sizing="stretch",
                opacity=1.0,
                layer="below",
                xanchor="left",
                yanchor="top",
            )

        if dataset_name == "luo2020":
            key_points = [
                "body_end",
                "cab_boom",
                "boom_arm",
                "arm_bucket",
                "bucket_end_left",
                "bucket_end_right",
                "arm_bucket",
            ]

            x = [target[key + "_x"] for key in key_points]
            y = [target[key + "_y"] for key in key_points]

            fig.add_trace(go.Scatter(x=x, y=y, text=key_points))

        elif dataset_name == "bang2020":
            for regions in target:
                x, y, w, h = regions["x"], regions["y"], regions["width"], regions["height"]
                fig.add_trace(
                    go.Scatter(
                        x=[x, x, x + w, x + w, x],
                        y=[y + h, y, y, y + h, y + h],
                        name=regions["phrase"],
                        fill="toself",
                        fillcolor="rgba(0,0,0,0)",
                        mode="lines",
                    )
                )

        elif dataset_name == "ACID":
            for obj in target:
                xmin, ymin, xmax, ymax = obj["bbox"]
                x = [xmin, xmin, xmax, xmax, xmin]
                y = [ymin, ymax, ymax, ymin, ymin]
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        name=obj["name"],
                        fill="toself",
                        fillcolor="rgba(0,0,0,0)",
                        mode="lines",
                        text=f"""<b>pose:</b> {obj["pose"]}<br>"""
                        + f"""<b>truncated:</b> {obj["truncated"]}"""
                        + f"""<br><b>difficult:</b> {obj["difficult"]}""",
                    )
                )

        elif dataset_name == "MOCS":
            import plotly.express as px

            colors = px.colors.qualitative.Plotly
            i = 0

            for obj in target:
                x, y, w, h = obj["bbox"]
                color = colors[i % len(colors)]

                fig.add_trace(
                    go.Scatter(
                        x=[x, x, x + w, x + w, x],
                        y=[y + h, y, y, y + h, y + h],
                        name=obj["category"],
                        mode="lines",
                        showlegend=False,
                        line=dict(color=color),
                    )
                )

                segmentation: np.ndarray = obj["segmentation"]
                show_legend = True
                for seg in segmentation:
                    fig.add_trace(
                        go.Scatter(
                            x=seg[:, 0],
                            y=seg[:, 1],
                            name=obj["category"],
                            fill="toself",
                            mode="lines",
                            showlegend=show_legend,
                            line=dict(color=color),
                        )
                    )

                    show_legend = False

                i = i + 1

        return fig, page, max_value, page, max_value

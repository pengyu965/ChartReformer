# from ChartSynthesizer import ChartSynthesizer
import json
import io
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# from MetaPropertyGenerator import MetaPropertyGenerator


class Replot:
    def __init__(self, param_dict, param_json_path):
        if not param_dict:
            with open(param_json_path, "r") as param_file:
                self.properties = json.load(param_file)

        self.src_data = pd.read_csv(io.StringIO(self.properties.pop("underlying_data").replace(" <0x0A> ", "\n")), sep="|")

        self.chart_title = self.properties.pop("chart_title")
        self.x_label = self.properties.pop("x_axis_title")
        self.y_label = self.properties.pop("y_axis_title")

        self.chart_type = self.properties["global_properties"].get("chart_type")

        self.linestyle_str_map = {'dense dotted': (0, (1, 1)), 'loose dotted':(0, (1, 3)), \
                                  'dense dashed': (0, (2, 1)), 'loose dashed':(0, (10, 3)) }

        self.fig = plt.figure(figsize=(10,8))

        self.ax = self.fig.add_subplot(111)

        if 'line' in self.chart_type:
            self._plot_line_chart()
        elif self.chart_type == 'grouped vertical bar':
            self._plot_grouped_vertical_bar()
        elif self.chart_type == 'stacked vertical bar':
            self._plot_stacked_vertical_bar()

    def get_data_table(self):
        return self.src_data.loc[:, :]

    # def sanitize_params(self, params: dict, default_properties=None) -> dict:
    #     chart_type = params["global_properties"].get("chart_type")

    #     if default_properties is None:
    #         default_properties = Replot.get_default_properties(self.src_data.shape[1]-1, chart_type)

    #     sanitized_params = Replot.replace_dict_recursively(default_properties, params)

    #     return sanitized_params

    # @staticmethod
    # def replace_dict_recursively(src: dict, target: dict):
    #     res = {}
    #     for key, value in src.items():
    #         target_val = target.get(key)
    #         if not target_val:
    #             target_val = value
    #         elif isinstance(target_val, dict):
    #             target_val = Replot.replace_dict_recursively(value, target_val)
    #         elif isinstance(target_val, list):
    #             target_val = target_val[:len(value)]

    #         res[key] = target_val
    #     return res

    # @staticmethod
    # def get_default_properties(legend_counts, chart_type="line"):
    #     default_properties = {
    #         "chart_title": "# Missing Chart Title",
    #         "x_axis_title": "# Missing x axis title",
    #         "y_axis_title": "# Missing y axis title",
    #         "global_properties": {
    #             "chart_type": "bar",
    #             "x_label_params": {
    #                 "fontname": "Arial Black",
    #                 "fontsize": "medium"
    #             },
    #             "y_label_params": {
    #                 "fontname": "Arial Black",
    #                 "fontsize": "medium"
    #             },
    #             "legend_params": {
    #                 "loc": 1,
    #                 "ncol": 3
    #             },
    #             "chart_title_params": {
    #                 "fontname": "Arial Black",
    #                 "fontsize": "large",
    #                 "rotation": 0
    #             },
    #             "x_tick_params": {
    #                 "axis": "x",
    #                 "which": "major",
    #                 "rotation": 45,
    #                 "labelsize": "small",
    #                 "labelfontfamily": "Arial Black"
    #             },
    #             "y_tick_params": {
    #                 "axis": "y",
    #                 "which": "major",
    #                 "rotation": 0,
    #                 "labelsize": "small",
    #                 "labelfontfamily": "Arial Black"
    #             },
    #             "grid_params": {
    #                 "visible": False
    #             }
    #         }}

    #     colors = Replot.repeat_to_length(MetaPropertyGenerator.COLOR_OPTIONS, legend_counts)
    #     if chart_type == "line":
    #         markers = Replot.repeat_to_length(MetaPropertyGenerator.MARKER_OPTIONS, legend_counts)
    #         line_styles = Replot.repeat_to_length(MetaPropertyGenerator.LINE_STYLES, legend_counts)
    #         default_properties.update(
    #             **{f"{chart_type}_properties": {"linestyles": line_styles,
    #                                             "markers": markers,
    #                                             "colors": colors
    #                                             }}
    #         )
    #     else:
    #         hatches = Replot.repeat_to_length(MetaPropertyGenerator.HATCH_OPTIONS, legend_counts)
    #         default_properties.update(
    #             **{f"{chart_type}_properties": {"hatches": hatches,
    #                                             "colors": colors}}
    #         )

    #     return default_properties

    # @staticmethod
    # def repeat_to_length(lst, length):
    #     if length <= 0:
    #         return []
    #     repeated_list = [lst[i % len(lst)] for i in range(length)]
    #     return repeated_list

    def _preprocess_table(self, all_bars):
        y_axis_label = all_bars.columns[0]
        for col in all_bars:
            try:
                all_bars[col] = pd.to_numeric(all_bars[col], downcast="integer").round(2)
            except ValueError:
                continue
        try:
            all_bars.columns = pd.to_numeric(all_bars.columns.to_series(), downcast="integer").to_frame("index").round(2)["index"].tolist()
            y_axis_label = all_bars.columns[0]
        except:
            try:
                all_bars.columns = [y_axis_label] + pd.to_numeric(all_bars.columns[1:].to_series(), downcast="integer").round(2).tolist()
            except:
                pass

        if pd.api.types.is_numeric_dtype(all_bars.dtypes[y_axis_label]):
            all_bars.sort_values(y_axis_label, inplace=True)
        return all_bars
    def _plot_line_chart(self):
        df = self._preprocess_table(self.get_data_table())
        first_col = df.columns[0]
        for idx, col in enumerate(df.columns[1:]):
            linestyle = self.properties["line_properties"]["linestyles"][idx]
            if linestyle in self.linestyle_str_map.keys():
                linestyle = self.linestyle_str_map[linestyle]
            self.ax.plot(df[first_col],
                         df[col],
                         label=col,
                         color=self.properties["line_properties"]["colors"][idx],
                         marker=self.properties["line_properties"]["markers"][idx],
                         linestyle=linestyle)
        self._plot_common_props()

    def _plot_grouped_vertical_bar(self):
        df = self._preprocess_table(self.get_data_table())
        emp = np.arange(df.shape[0])
        bar_padding = 0.05
        bar_width = (1 / (df.shape[1] + 1)) - bar_padding

        first_col = df.columns[0]
        bar_property_keyname = self.chart_type+"_properties"
        for idx, col in enumerate(df.columns[1:]):
            base_positions = emp - (df.shape[1] * bar_width / 2) + (idx * bar_width)
            kwargs = dict(width=bar_width,
                          label=col,
                          color=self.properties[bar_property_keyname]["colors"][idx],
                          hatch=self.properties[bar_property_keyname]["hatches"][idx]
                          )
            args = [base_positions]
            args.append(df[[col]].rename(columns={col: "y"})["y"])
            self.ax.bar(*args, **kwargs)

        leftmost_positions = emp - (df.shape[1] * bar_width / 2)
        rightmost_positions = emp + (df.shape[1] * bar_width / 2) - bar_width
        center_positions = (leftmost_positions + rightmost_positions) / 2

        self.ax.set_xticks(center_positions)
        self.ax.set_xticklabels(df[first_col])
        self._plot_common_props()

    def _plot_stacked_vertical_bar(self):
        df = self._preprocess_table(self.get_data_table()).T.reset_index()
        df.columns = df.iloc[0, :].values
        df = df.drop(index=0).reset_index(drop=True)
        emp = np.arange(df.shape[0])
        bar_padding = 0.05
        bar_width = (1 / (df.shape[0] + 1)) - bar_padding
        x_ticks = df.columns[1:]
        bar_property_keyname = self.chart_type+"_properties"

        bottoms = np.zeros(df.shape[1] - 1)
        for idx, row in df.iterrows():
            self.ax.bar(df.columns[1:], row.iloc[1:].values,
                        label=row.iloc[0],
                        bottom=bottoms,
                        hatch=self.properties[bar_property_keyname]['hatches'][idx],
                        color=self.properties[bar_property_keyname]['colors'][idx],
                        width=bar_width)
            bottoms += row.iloc[1:]

        leftmost_positions = emp - (df.shape[1] * bar_width / 2)
        rightmost_positions = emp + (df.shape[1] * bar_width / 2) - bar_width
        center_positions = (leftmost_positions + rightmost_positions) / 2

        self.ax.set_xticks(center_positions)
        self.ax.set_xticklabels(x_ticks)
        self._plot_common_props()

    def _plot_common_props(self):
        if self.x_label and 'x_label_params' in self.properties['global_properties'] and self.properties['global_properties']['x_label_params']:
            self.ax.set_xlabel(self.x_label, **self.properties['global_properties']['x_label_params'])
        if self.y_label and 'y_label_params' in self.properties['global_properties'] and self.properties['global_properties']['y_label_params']:
            self.ax.set_ylabel(self.y_label, **self.properties['global_properties']['y_label_params'])
        if self.chart_title and 'chart_title_params' in self.properties['global_properties'] and self.properties['global_properties']['chart_title_params']:
            self.ax.set_title(self.chart_title, **self.properties['global_properties']['chart_title_params'])
        if 'legend_params' in self.properties['global_properties'] and self.properties['global_properties']['legend_params']:
            self.ax.legend(**self.properties['global_properties']['legend_params'])
        if 'grid_params' in self.properties['global_properties'] and self.properties['global_properties']['grid_params']:
            self.ax.grid(**self.properties['global_properties']['grid_params'])
        if 'x_tick_params' in self.properties['global_properties'] and self.properties['global_properties']['x_tick_params']:
            self.ax.tick_params(**self.properties['global_properties']['x_tick_params'])
        if 'y_tick_params' in self.properties['global_properties'] and self.properties['global_properties']['y_tick_params']:
            self.ax.tick_params(**self.properties['global_properties']['y_tick_params'])

        self.fig.tight_layout()


if __name__ == "__main__":
    import shutil
    from tqdm import tqdm
    import os

    gt_image_path = '../../data/illustration/images/'
    gt_prompt_path = '../../data/illustration/prompts/'
    table_pre_path = '../../data/illustration/results/'

    vis_save_path = "../../data/illustration/vis_res/"
    os.makedirs(vis_save_path, exist_ok=True)

    successed_num = 0
    for file in tqdm(os.listdir(table_pre_path)):
        chart_id = file[:-5]

        try:
            params_dict = {}
            replot = Replot(params_dict,table_pre_path + file)
            replot.fig.savefig(vis_save_path + chart_id + "_edited" + ".jpg")
            successed_num += 1
        except:
            continue
        shutil.copy(gt_image_path + chart_id + ".jpg", vis_save_path + chart_id + ".jpg")
        shutil.copy(gt_prompt_path + chart_id + ".txt", vis_save_path + chart_id + ".txt")

    print("Successed: ", successed_num, "Failed: ", len(os.listdir(table_pre_path)) - successed_num, "Total: ", len(os.listdir(table_pre_path)))

import ipycytoscape
import ipywidgets as widgets
import numpy as np
import torch
from IPython.display import HTML
from IPython.display import display, clear_output
from torch import nn

from gm.hmoe.hierarchical_moe import HierarchicalMoE


class CytoscapeHMoEVisualizer:
    def __init__(self, llm_wrapper: nn.Module, scenario, experts_count: int, top_k: int = 2):
        """
        Финальный визуализатор HMoE: только эксперты и суммы L{X}

        Args:
            scenario: объект Scenario с вычислениями
            experts_count: общее количество экспертов
            top_k: количество выбираемых экспертов на каждом уровне
        """
        self.hmoe: HierarchicalMoE = llm_wrapper.model
        self.output_head: nn.Module = llm_wrapper.output_head
        self.scenario = scenario
        self.experts_count = experts_count
        self.top_k = top_k

        # Текущие данные
        self.current_level = 0
        self.current_batch = 0

        # Размеры
        self.x_gap = 500  # расстояние между колонками
        self.y_gap = 200  # расстояние между строками

        # Цвета для ребер по k-индексу
        self.edge_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']

        # Создаем виджеты
        self.create_widgets()

        # Инициализируем данные
        self.load_data()

        # Создаем граф
        self.create_graph()

        # Отображаем
        self.display()

    def create_widgets(self):
        """Создание виджетов управления"""
        # Панель управления
        self.level_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=max(0, len(self.scenario.chains) - 1),
            description='Уровень:',
            style={'description_width': 'initial'}
        )

        self.batch_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=0,
            description='Батч:',
            style={'description_width': 'initial'}
        )

        # Кнопки
        self.update_btn = widgets.Button(description='🔄 Обновить', button_style='primary')
        self.reset_btn = widgets.Button(description='↺ Сбросить', button_style='warning')
        self.recalc_btn = widgets.Button(description='🧮 Пересчитать', button_style='success')

        # Настройки
        self.show_weights = widgets.Checkbox(
            value=True,
            description='Показать веса на ребрах',
            indent=False
        )

        # Редакторы
        self.selected_info = widgets.HTML(value="<b>Выбранный элемент:</b> -")

        self.value_editor = widgets.FloatSlider(
            min=0.0, max=1.0, step=0.01, value=0.5,
            description='Значение:', disabled=True
        )

        # Редактор матрицы для L{X}
        self.matrix_editor = widgets.Textarea(
            value='',
            placeholder='JSON array or nested lists (e.g. [0.1, 0.2, ...])',
            layout=widgets.Layout(width='420px', height='160px'),
            disabled=True
        )
        self.save_matrix_btn = widgets.Button(description='💾 Save matrix', button_style='primary', disabled=True)
        self.save_matrix_btn.on_click(lambda _: self._save_matrix_from_editor())

        # Отображение top-k из output_head
        self.topk_output = widgets.HTML(value="")  # html block showing top-k indices + probs

        # Собираем панели (единственный edit_panel — ранее у вас было 2)
        self.edit_panel = widgets.VBox([
            widgets.HTML("<h4>Редактирование</h4>"),
            self.selected_info,
            self.value_editor,
            widgets.HTML("<b>Edit latent (L{c}):</b>"),
            self.matrix_editor,
            widgets.HBox([self.save_matrix_btn, widgets.Label(" "), widgets.Label("Top-5 (output head):")]),
            self.topk_output
        ], layout=widgets.Layout(border='1px solid #ddd', padding='10px'))

        # Панель управления (control_panel)
        self.control_panel = widgets.VBox([
            widgets.HTML("<h4>Параметры</h4>"),
            widgets.HBox([self.level_slider, self.batch_slider]),
            self.show_weights,
            widgets.HBox([self.update_btn, self.reset_btn, self.recalc_btn])
        ], layout=widgets.Layout(border='1px solid #ddd', padding='10px'))

        # Граф контейнер и таблица
        self.graph_container = widgets.VBox()
        self.table_container = widgets.VBox()

        # Подписки на события
        self.level_slider.observe(lambda _: self.on_level_changed(), 'value')
        self.batch_slider.observe(lambda _: self.on_batch_changed(), 'value')
        self.update_btn.on_click(lambda _: self.update_graph())
        self.reset_btn.on_click(lambda _: self.reset_data())
        self.recalc_btn.on_click(lambda _: self.recalculate())
        self.show_weights.observe(lambda _: self.update_graph(), 'value')
        self.value_editor.observe(lambda c: self.update_selected_value(c), 'value')

    def update_selected_value(self, change):
        """Обновление значения выбранного элемента (узел эксперт или ребро с весом)"""
        # Если выбран узел-эксперт
        if hasattr(self, 'selected_node_id') and self.selected_node_id:
            for node in self.cytoscape.graph.nodes:
                if node['data']['id'] == self.selected_node_id:
                    if self.selected_node_id.startswith('E'):
                        expert_id = node['data']['expert']
                        self.expert_values[expert_id] = change['new']
                        node['data']['value'] = change['new']
                    break

        # Если выбран ребро с весом (есть поле 'weight')
        elif hasattr(self, 'selected_edge_id') and self.selected_edge_id:
            for edge in self.cytoscape.graph.edges:
                if edge['data']['id'] == self.selected_edge_id:
                    data = edge['data']
                    if 'weight' in data:
                        # обновляем сам объект ребра
                        data['weight'] = float(change['new'])
                        data['label'] = f'{change["new"]:.2f}'

                        # парсим source вида "E{pos}_{k}"
                        src = data.get('source', '')
                        if isinstance(src, str) and src.startswith('E'):
                            # пример src = "E3_1"
                            try:
                                parts = src[1:].split('_')
                                pos = int(parts[0])
                                k = int(parts[1])
                                # сохраняем в editable_weights
                                self.editable_weights[pos, k] = float(change['new'])
                                # отметим последнее изменение по позиции pos
                                self.last_modified = (self.current_level, int(pos))
                            except Exception:
                                pass
                    break

        # Перерисовываем граф и пересчитываем суммы
        try:
            self.cytoscape.graph.redraw()
        except Exception:
            # Иногда redraw недоступен — просто пересоздаем граф
            self.create_graph()
        self.update_sums()

    def load_data(self):
        """Загрузка данных (поддерживаем формат: scenario.latents[level][position] -> (batch, time_steps, d_model))"""
        if len(self.scenario.chains) == 0:
            return

        self.last_modified = None

        batch_size = self.scenario.chains[self.current_level].shape[0]
        self.batch_slider.max = max(0, batch_size - 1)

        # Получаем chains/weights (как раньше)
        self.chains_data = self.scenario.chains[self.current_level][self.current_batch].detach().cpu().numpy()
        self.weights_data = self.scenario.weights[self.current_level][self.current_batch].detach().cpu().numpy()

        # latents: ожидаем список по позициям: scenario.latents[level][pos] -> tensor (batch, time_steps, d_model)
        self.latents = []
        try:
            level_latents = self.scenario.latents[self.current_level]
            # если level_latents — список/iterable по позициям
            for pos_tensor in level_latents:
                # pos_tensor may be torch.Tensor(shape=(batch, T, D))
                if isinstance(pos_tensor, torch.Tensor):
                    self.latents.append(pos_tensor.detach().cpu().numpy())
                else:
                    # если уже numpy
                    self.latents.append(np.asarray(pos_tensor))
            # теперь self.latents[pos] -> ndarray (batch, T, D)
        except Exception:
            # fallback: старый формат — список снэпшотов shape (batch, positions, features)
            try:
                self.latents = [snap[self.current_batch].detach().cpu().numpy() for snap in
                                self.scenario.latents[self.current_level]]
            except Exception:
                self.latents = []

        # Сохраняем оригиналы при первом запуске
        if not hasattr(self, 'original_chains'):
            self.original_chains = self.chains_data.copy()
            self.original_weights = self.weights_data.copy()
            self.original_latents = [arr.copy() for arr in self.latents]

        # Инициализируем редактируемые данные
        self.editable_chains = self.chains_data.copy()
        self.editable_weights = self.weights_data.copy()
        self.editable_latents = [arr.copy() for arr in self.latents]

        # Словарь для значений экспертов
        self.expert_values = {}
        unique_experts = np.unique(self.editable_chains)
        for expert in unique_experts:
            self.expert_values[int(expert)] = 0.5

        # Создаем таблицу для редактирования
        self.create_editable_table()

    def create_editable_table(self):
        """Создание таблицы для редактирования цепочек и весов"""
        if self.editable_chains is None:
            return

        positions_count = len(self.editable_chains)

        # Создаем GridBox для таблицы
        table_widgets = []

        # Заголовки
        table_widgets.append(widgets.HTML("<b>Поз.</b>"))
        for k in range(self.top_k):
            table_widgets.append(widgets.HTML(f"<b>Эксперт {k}</b>"))
        for k in range(self.top_k):
            table_widgets.append(widgets.HTML(f"<b>Вес {k}</b>"))

        # Данные
        for pos in range(positions_count):
            # Номер позиции
            table_widgets.append(widgets.Label(f"{pos}"))

            # Поля для экспертов
            for k in range(self.top_k):
                text = widgets.Text(
                    value=str(int(self.editable_chains[pos, k])),
                    layout=widgets.Layout(width='60px')
                )
                text.observe(lambda change, p=pos, _k=k:
                             self.on_chain_changed(change, p, _k), 'value')
                table_widgets.append(text)

            # Поля для весов
            for k in range(self.top_k):
                text = widgets.Text(
                    value=f"{self.editable_weights[pos, k]:.3f}",
                    layout=widgets.Layout(width='60px')
                )
                text.observe(lambda change, p=pos, _k=k:
                             self.on_weight_changed(change, p, _k), 'value')
                table_widgets.append(text)

        # Создаем GridBox
        grid = widgets.GridBox(
            children=table_widgets,
            layout=widgets.Layout(
                grid_template_columns=f'auto ' + ' '.join(['auto'] * (self.top_k * 2)),
                grid_gap='2px',
                padding='10px'
            )
        )

        # Контейнер с заголовком и скроллом
        self.table_container.children = [
            widgets.HTML("<h4>Таблица редактирования (двойной клик для редактирования)</h4>"),
            widgets.Box([grid], layout=widgets.Layout(
                overflow_x='auto',
                overflow_y='auto',
                max_height='250px',
                border='1px solid #ddd',
                padding='10px'
            ))
        ]

    def create_graph(self):
        """Создание/обновление графа: widget создаётся один раз; далее только обновление содержимого."""
        if self.editable_chains is None:
            return

        positions_count = len(self.editable_chains)
        nodes = []
        edges = []

        inter_gap = 50  # промежуток между колонками (визуальный)

        # 1) Узлы экспертов — в табличной сетке
        for r in range(self.top_k):
            for c in range(positions_count):
                expert_id = int(self.editable_chains[c, r])
                weight = float(self.editable_weights[c, r])

                node_id = f"E{c}_{r}"
                x = float(c * (self.x_gap + inter_gap))
                y = float(r * self.y_gap)
                nodes.append({
                    'data': {
                        'id': node_id,
                        'label': f"E{expert_id}",
                        'expert': int(expert_id),
                        'pos_index': int(c),
                        'k_index': int(r),
                        'weight': float(weight),
                        'value': float(self.expert_values.get(expert_id, 0.5))
                    },
                    'position': {'x': x, 'y': y},
                    'locked': True,
                    'classes': 'table-node'
                })

        # 2) Промежуточные (latent / суммы)
        for j in range(positions_count + 1):
            inter_x = float(j * (self.x_gap + inter_gap) - self.x_gap / 2)
            inter_y = ((self.top_k - 1) * self.y_gap) / 2.0  # центр по вертикали
            sum_id = f"L{j}"
            sum_value = 0.0
            if j > 0:
                for r in range(self.top_k):
                    expert = int(self.editable_chains[j - 1, r])
                    w = float(self.editable_weights[j - 1, r])
                    sum_value += w * self.expert_values.get(expert, 0.5)

            nodes.append({
                'data': {'id': sum_id, 'label': f"L{j}", 'pos_index': int(j), 'value': float(sum_value)},
                'position': {'x': inter_x, 'y': inter_y},
                'locked': True,
                'classes': 'inter-node'
            })

        # 3) Ребра
        for c in range(positions_count):
            for r in range(self.top_k):
                expert_node = f"E{c}_{r}"
                left_latent = f"L{c}"
                right_latent = f"L{c + 1}"

                edges.append({
                    'data': {
                        'id': f"w_{expert_node}_to_{right_latent}",
                        'source': expert_node,
                        'target': right_latent,
                        'weight': float(self.editable_weights[c, r]),
                        'strval': f'{self.editable_weights[c, r]:.3f}',
                        'k_index': int(r)
                    },
                    'classes': f'edge-k{r}'
                })

                edges.append({
                    'data': {
                        'id': f"f_{left_latent}_to_{expert_node}",
                        'source': left_latent,
                        'target': expert_node,
                        'k_index': int(r)
                    },
                    'classes': 'edge-from-sum'
                })

        graph_json = {'nodes': nodes, 'edges': edges}

        # Создаём виджет один раз; при последующих вызовах просто обновляем graph
        if not hasattr(self, 'cytoscape') or self.cytoscape is None:
            self.cytoscape = ipycytoscape.CytoscapeWidget(
                layout={'name': 'preset'},
                style={'width': '100%', 'height': '500px', 'background': '#f8f9fa', 'border': '2px solid #dee2e6'}
            )
            self.cytoscape.graph.add_graph_from_json(graph_json)
            # обработчики один раз
            self.cytoscape.on('node', 'click', self.on_node_click)
            self.cytoscape.on('edge', 'click', self.on_edge_click)
        else:
            # обновляем существующий граф (без пересоздания widget)
            try:
                self.cytoscape.graph.clear()
                self.cytoscape.graph.add_graph_from_json(graph_json)
            except Exception:
                # в редком случае, если clear/add вызывает некорректное поведение, пересоздаём widget
                self.cytoscape = ipycytoscape.CytoscapeWidget(
                    layout={'name': 'preset'},
                    style={'width': '100%', 'height': '500px', 'background': '#f8f9fa', 'border': '2px solid #dee2e6'}
                )
                self.cytoscape.graph.add_graph_from_json(graph_json)
                self.cytoscape.on('node', 'click', self.on_node_click)
                self.cytoscape.on('edge', 'click', self.on_edge_click)

        # Применяем стили и обновляем контейнер
        self.apply_styles()
        self.graph_container.children = [
            widgets.HTML(
                f"<h4>Визуализация HMoE | Уровень {self.current_level}, Батч {self.current_batch} | "
                f"Позиций: {positions_count}, Top-K: {self.top_k}</h4>"
            ),
            self.cytoscape
        ]

    def apply_styles(self):
        """Применение стилей — добавлены table-node и inter-node, а также аккуратные стрелки"""
        stylesheet = [
            {
                'selector': '.table-node',
                'style': {
                    'background-color': '#3498db',
                    'label': 'data(label)',
                    'width': 40,
                    'height': 40,
                    'color': 'white',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'font-weight': 'bold',
                    'border-width': 2,
                    'border-color': '#2980b9'
                }
            },
            {
                'selector': '.inter-node',
                'style': {
                    'background-color': '#2ecc71',
                    'label': 'data(label)',
                    'width': 46,
                    'height': 46,
                    'color': 'white',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'shape': 'diamond',
                    'border-width': 2,
                    'border-color': '#27ae60'
                }
            },
            {
                'selector': '.selected',
                'style': {
                    'border-color': '#e74c3c',
                    'border-width': 4,
                    'background-color': '#e74c3c'
                }
            },
            {
                'selector': '.edge-selected',
                'style': {
                    'line-color': '#e74c3c',
                    'width': 6,
                    'target-arrow-color': '#e74c3c'
                }
            }
        ]

        # стили для ребер с весом (edge-k*)
        for k in range(self.top_k):
            color = self.edge_colors[k % len(self.edge_colors)]
            stylesheet.append({
                'selector': f'.edge-k{k}',
                'style': {
                    'line-color': color,
                    # толщина мапируется от веса; если веса нет, будет 2
                    'width': 'mapData(weight, 0, 1, 2, 8)',
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle',
                    'target-arrow-color': color,
                    'arrow-scale': 1.0,
                    'label': 'data(strval)',
                    'font-size': '10px',
                    'text-rotation': 'autorotate',
                    'text-margin-y': -8
                }
            })

        # стиль для визуальных связей от latent -> expert (без веса, минимальные стрелки)
        stylesheet.append({
            'selector': '.edge-from-sum',
            'style': {
                'line-color': '#95a5a6',
                'width': 2,
                'curve-style': 'bezier',
                'target-arrow-shape': 'triangle',
                'target-arrow-color': '#95a5a6',
                'arrow-scale': 0.9,
                'label': ''
            }
        })

        self.cytoscape.set_style(stylesheet)

    def update_graph(self):
        """Обновление графа"""
        self.create_graph()

    # Обработчики событий таблицы
    def on_chain_changed(self, change, pos, k):
        """Изменение цепочки в таблице"""
        try:
            new_expert = int(change['new'])
            if 0 <= new_expert < self.experts_count:
                self.editable_chains[pos, k] = new_expert
                # отметим последнее изменение: текущий уровень и позиция pos
                self.last_modified = (self.current_level, int(pos))
                self.update_graph()
        except ValueError:
            pass

    def on_weight_changed(self, change, pos, k):
        """Изменение веса в таблице"""
        try:
            new_weight = float(change['new'])
            if 0.0 <= new_weight <= 1.0:
                self.editable_weights[pos, k] = new_weight
                # отметим последнее изменение
                self.last_modified = (self.current_level, int(pos))
                self.update_graph()
        except ValueError:
            pass

    # Обработчики событий графа
    def on_node_click(self, event):
        node_data = event.get('data') if isinstance(event, dict) else None
        if node_data is None:
            return
        node_id = node_data.get('id')
        if node_id is None:
            return

        # снимем старое выделение
        try:
            if hasattr(self, 'selected_node_id') and self.selected_node_id:
                try:
                    self.cytoscape.set_class(self.selected_node_id, 'selected', False)
                except Exception:
                    pass
        except Exception:
            pass

        # выделим новый
        try:
            self.cytoscape.set_class(node_id, 'selected', True)
        except Exception:
            pass
        self.selected_node_id = node_id

        # очистим выбранное ребро, если было
        if hasattr(self, 'selected_edge_id') and self.selected_edge_id:
            try:
                self.cytoscape.set_class(self.selected_edge_id, 'edge-selected', False)
            except Exception:
                pass
            self.selected_edge_id = None

        # Обработка экспертов
        if node_id.startswith('E'):
            expert = node_data.get('expert', None)
            pos = node_data.get('pos_index', node_data.get('position', 'N/A'))
            k_idx = node_data.get('k_index', 'N/A')
            weight = node_data.get('weight', 0.0)
            value = node_data.get('value', 0.0)

            info = f"""
            <b>Эксперт E{expert}</b><br>
            Позиция: {pos}<br>
            k-index: {k_idx}<br>
            Вес: {float(weight):.3f}<br>
            Значение: {float(value):.3f}
            """
            self.value_editor.disabled = False
            try:
                self.value_editor.value = float(value)
            except Exception:
                self.value_editor.value = 0.5

            # отключаем matrix editor
            self.matrix_editor.disabled = True
            self.matrix_editor.value = ''
            self.save_matrix_btn.disabled = True
            self.topk_output.value = ''

        # Обработка сумм L{X}
        elif node_id.startswith('L'):
            pos = node_data.get('pos_index', node_data.get('position', 'N/A'))
            value = node_data.get('value', 0.0)
            info = f"""
                    <b>Сумма L{pos}</b><br>
                    Значение: {float(value):.3f}
                    """
            self.value_editor.disabled = True

            # Попытка найти latent для этой позиции: ожидаем self.latents[pos] -> (batch, time_steps, d_model)
            latent_text = ''
            latent_matrix = None
            try:
                c = int(pos)
                if c < 0 or c >= len(self.latents):
                    raise IndexError("position out of range")
                arr = np.asarray(self.latents[c])  # (batch, T, D) or maybe already (T, D)
                if arr.ndim == 3:
                    # берём для текущего батча: (T, D)
                    latent_matrix = arr[self.current_batch]
                elif arr.ndim == 2:
                    # уже (T, D)
                    latent_matrix = arr
                elif arr.ndim == 1:
                    # в крайнем случае — сделаем (T=1, D)
                    latent_matrix = arr.reshape(1, -1)
                else:
                    raise ValueError("unexpected latent array ndim")
                import json
                latent_text = json.dumps(latent_matrix.tolist(), indent=2)
                # Запомним ссылку: уровень + позиция (pos) — при сохранении обновим именно этот position
                self._matrix_latent_ref = ('pos', c)
            except Exception:
                latent_text = '[]'
                self._matrix_latent_ref = None

            # Включаем редактор
            self.matrix_editor.value = latent_text
            self.matrix_editor.disabled = False
            self.save_matrix_btn.disabled = (self._matrix_latent_ref is None)

            # Если есть output_head и latent_matrix — вычислим top-5 для каждого временного шага
            if latent_matrix is not None and latent_matrix.size != 0 and hasattr(self,
                                                                                 'output_head') and self.output_head is not None:
                try:
                    topk_html = self.compute_topk_html(latent_matrix, topk=5, max_steps_display=50)
                except Exception as e:
                    topk_html = f"<b>Top-k error:</b> {e}"
                self.topk_output.value = topk_html
            else:
                self.topk_output.value = ''

        else:
            # прочие узлы
            self.value_editor.disabled = True
            self.matrix_editor.disabled = True
            self.save_matrix_btn.disabled = True
            self.topk_output.value = ''

        self.selected_info.value = info

    def on_edge_click(self, event):
        """Клик по ребру — корректно показывает вес и включает редактирование веса."""
        edge_data = event.get('data') if isinstance(event, dict) else None
        if edge_data is None:
            return

        edge_id = edge_data.get('id')
        if edge_id is None:
            return

        # снять предыдущее выделение ребра
        try:
            if hasattr(self, 'selected_edge_id') and self.selected_edge_id:
                try:
                    self.cytoscape.set_class(self.selected_edge_id, 'edge-selected', False)
                except Exception:
                    pass
        except Exception:
            pass

        # удалить выделение узла, если было
        try:
            if hasattr(self, 'selected_node_id') and self.selected_node_id:
                try:
                    self.cytoscape.set_class(self.selected_node_id, 'selected', False)
                except Exception:
                    pass
                self.selected_node_id = None
        except Exception:
            pass

        # установить выделение на ребро
        try:
            self.cytoscape.set_class(edge_id, 'edge-selected', True)
        except Exception:
            pass
        self.selected_edge_id = edge_id

        weight = edge_data.get('weight', None)
        k_idx = edge_data.get('k_index', 'N/A')

        if weight is None:
            info = f"<b>Ребро</b><br>Вес: N/A<br>k-index: {k_idx}"
            self.value_editor.disabled = True
        else:
            info = f"<b>Ребро</b><br>Вес: {float(weight):.3f}<br>k-index: {k_idx}"
            self.value_editor.disabled = False
            try:
                self.value_editor.value = float(weight)
            except Exception:
                self.value_editor.value = 0.0

        self.selected_info.value = info

    def on_level_changed(self):
        """Смена уровня: корректно обновляем current_level и current_batch, затем перезагружаем данные."""
        new_lvl = int(self.level_slider.value)
        self.current_level = new_lvl

        # Обновляем batch_slider.max внутри load_data; но сначала убедимся, что current_batch валиден
        # Сбрасываем батч на 0 при смене уровня (безопаснее)
        self.current_batch = 0
        # синхронизируем слайдер
        try:
            self.batch_slider.unobserve_all()
        except Exception:
            pass
        self.batch_slider.value = 0
        # восстанавливаем обработчики наблюдения (чтобы избежать дублирования)
        self.batch_slider.observe(lambda _: self.on_batch_changed(), 'value')

        # Перезагружаем данные и обновляем граф
        self.load_data()
        self.update_graph()

    def on_batch_changed(self):
        self.current_batch = self.batch_slider.value
        self.load_data()
        self.update_graph()

    def reset_data(self):
        """Сброс данных к оригинальным"""
        if hasattr(self, 'original_chains'):
            self.editable_chains = self.original_chains.copy()
            self.editable_weights = self.original_weights.copy()

            # Сбрасываем значения экспертов
            unique_experts = np.unique(self.editable_chains)
            for expert in unique_experts:
                self.expert_values[int(expert)] = 0.5

            # Обновляем таблицу и граф
            self.create_editable_table()
            self.update_graph()

    def _save_matrix_from_editor(self):
        """Парсит JSON из matrix_editor и сохраняет в self.latents / editable_latents и в scenario.latents.
           Ожидается 2D матрица (time_steps, d_model) — запись в self.latents[pos][batch]"""
        import json
        txt = self.matrix_editor.value
        try:
            parsed = json.loads(txt)
        except Exception as e:
            self.selected_info.value = f"<b>Ошибка парсинга JSON:</b> {e}"
            return

        if not hasattr(self, '_matrix_latent_ref') or self._matrix_latent_ref is None:
            self.selected_info.value = "<b>Нет ссылки на latent для сохранения.</b>"
            return

        ref_type, pos = self._matrix_latent_ref
        if ref_type != 'pos':
            self.selected_info.value = "<b>Неподдерживаемая ссылка на latent.</b>"
            return

        try:
            new_arr = np.asarray(parsed)
        except Exception as e:
            self.selected_info.value = f"<b>Ошибка преобразования в массив:</b> {e}"
            return

        if new_arr.ndim != 2:
            self.selected_info.value = "<b>Ожидается 2D матрица (time_steps x d_model).</b>"
            return

        # 1) Обновим локальные self.latents[pos][batch] = new_arr
        try:
            if pos < 0 or pos >= len(self.latents):
                raise IndexError("position out of range")
            arr = self.latents[pos]  # expected (batch, T, D) or (T, D)
            if arr.ndim == 3:
                self.latents[pos][self.current_batch] = new_arr
            elif arr.ndim == 2:
                # replace whole array for this pos
                self.latents[pos] = new_arr
            else:
                # try to broadcast/replace
                self.latents[pos] = new_arr
            self.last_modified = (self.current_level, int(pos))
        except Exception:
            try:
                # попытка конвертации и повторная запись
                self.latents[pos] = np.asarray(self.latents[pos])
                if self.latents[pos].ndim == 3:
                    self.latents[pos][self.current_batch] = new_arr
                else:
                    self.latents[pos] = new_arr
                self.last_modified = (self.current_level, int(pos))
            except Exception:
                pass

        # 2) editable_latents
        try:
            if len(self.editable_latents) <= pos:
                # расширим
                while len(self.editable_latents) <= pos:
                    self.editable_latents.append(None)
            if self.editable_latents[pos] is None:
                self.editable_latents[pos] = new_arr
            else:
                arr2 = np.asarray(self.editable_latents[pos])
                if arr2.ndim == 3:
                    arr2[self.current_batch] = new_arr
                    self.editable_latents[pos] = arr2
                else:
                    self.editable_latents[pos] = new_arr
        except Exception:
            pass

        # 3) Запишем в scenario.latents, если ожидаемый формат: scenario.latents[level][pos] tensor (batch, T, D)
        try:
            lvl = self.current_level
            b = self.current_batch
            t = self.scenario.latents[lvl][pos]  # tensor (batch, T, D)
            if isinstance(t, torch.Tensor):
                t_clone = t.clone()
                new_t = torch.from_numpy(new_arr).to(dtype=t_clone.dtype, device=t_clone.device)
                # new_t shape (T, D) -> assign to t_clone[b]
                if new_t.ndim == 2 and t_clone.ndim == 3:
                    t_clone[b] = new_t
                    self.scenario.latents[lvl][pos] = t_clone
                else:
                    # attempt flatten/reshape fallback
                    try:
                        flat = new_t.flatten()[: t_clone[b].numel()]
                        reshaped = flat.view_as(t_clone[b])
                        t_clone[b] = reshaped
                        self.scenario.latents[lvl][pos] = t_clone
                    except Exception:
                        # несоответствие формы — не записываем
                        self.selected_info.value = "<b>Матрица локально сохранена, но запись в scenario не удалась (несовместимые формы).</b>"
            else:
                # если scenario.latents хранит numpy, попробуем записать
                try:
                    sarr = np.asarray(self.scenario.latents[lvl][pos])
                    if sarr.ndim == 3:
                        sarr[b] = new_arr
                        self.scenario.latents[lvl][pos] = sarr
                    else:
                        self.scenario.latents[lvl][pos] = new_arr
                except Exception:
                    pass
        except Exception:
            # не критично
            pass

        self.selected_info.value = "<b>Матрица сохранена (локально).</b>"

        # Обновим top-k отображение для текущей latent
        try:
            latent_array = np.asarray(new_arr)
            if hasattr(self, 'output_head') and self.output_head is not None:
                self.topk_output.value = self.compute_topk_html(latent_array, topk=5, max_steps_display=50)
        except Exception:
            pass

    def _apply_edits_to_scenario(self):
        """Записать editable_* обратно в self.scenario (попытаться аккуратно по типам)."""
        lvl = self.current_level
        b = self.current_batch

        # Преобразуем в тензоры
        chains_t = torch.tensor(self.editable_chains)
        weights_t = torch.tensor(self.editable_weights, dtype=torch.float32)

        try:
            # Если сценарий хранит батчи внутри тензора: shape (batch, positions, top_k)
            if isinstance(self.scenario.chains[lvl], torch.Tensor):
                self.scenario.chains[lvl][b] = chains_t
            else:
                # Часто scenario.chains[lvl] — список/массив батчей
                self.scenario.chains[lvl][b] = chains_t
        except Exception:
            # fallback: заменим целиком уровень (без попыток сохранять device)
            try:
                self.scenario.chains[lvl] = torch.stack([chains_t])
            except Exception:
                self.scenario.chains[lvl] = [chains_t]

        try:
            if isinstance(self.scenario.weights[lvl], torch.Tensor):
                self.scenario.weights[lvl][b] = weights_t
            else:
                self.scenario.weights[lvl][b] = weights_t
        except Exception:
            try:
                self.scenario.weights[lvl] = torch.stack([weights_t])
            except Exception:
                self.scenario.weights[lvl] = [weights_t]

    def _load_from_scenario(self):
        """Загрузить текущие arrays из self.scenario в editable_* и latents."""
        lvl = self.current_level
        b = self.current_batch

        # Берём тензоры аккуратно и приводим к numpy
        chains_obj = self.scenario.chains[lvl][b]
        weights_obj = self.scenario.weights[lvl][b]

        # Если это list/ndarray — привести в torch -> numpy
        if isinstance(chains_obj, np.ndarray):
            chains_np = chains_obj.copy()
        else:
            chains_np = chains_obj.detach().cpu().numpy()

        if isinstance(weights_obj, np.ndarray):
            weights_np = weights_obj.copy()
        else:
            weights_np = weights_obj.detach().cpu().numpy()

        self.chains_data = chains_np
        self.weights_data = weights_np

        # latents: если есть структура self.scenario.latents[lvl] — ожидаем список тензоров (по слоям)
        try:
            self.latents = [l[b].detach().cpu().numpy() for l in self.scenario.latents[self.current_level]]
        except Exception:
            # безопасный fallback
            self.latents = []

        # Обновляем editable
        self.editable_chains = self.chains_data.copy()
        self.editable_weights = self.weights_data.copy()

        # Обновляем оригиналы (чтобы reset работал)
        self.original_chains = self.chains_data.copy()
        self.original_weights = self.weights_data.copy()
        self.original_latents = self.latents.copy()

    def compute_topk(self, latent_array: np.ndarray, topk: int = 5):
        """
        Возвращает (indices, scores) topk по output_head для данного latent_array.
        - latent_array: 1D (features,) или 2D (time_steps, features).
        - Для 1D возвращает lists length=time_steps=1; для 2D возвращает arrays shape (T, topk).
        """
        import torch.nn.functional as F

        arr = np.asarray(latent_array)
        if arr.ndim == 0:
            raise ValueError("latent_array must be 1D or 2D")

        device = None
        if hasattr(self, 'output_head') and any(True for _ in self.output_head.parameters()):
            device = next(self.output_head.parameters()).device
        else:
            device = torch.device('cpu')

        # Make 2D: (T, D)
        if arr.ndim == 1:
            bat = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).to(device=device, dtype=torch.float32)
        else:
            bat = torch.from_numpy(arr.astype(np.float32)).to(device=device, dtype=torch.float32)  # (T, D)

        with torch.no_grad():
            logits = self.output_head(bat)  # expect (B=T or 1, V)
            if isinstance(logits, tuple):
                logits = logits[0]
            # logits shape: (B, V)
            probs = F.softmax(logits, dim=-1)
            k = min(topk, probs.size(-1))
            top = torch.topk(probs, k, dim=-1)
            indices = top.indices.cpu().numpy()  # shape (B, k)
            scores = top.values.cpu().numpy()  # shape (B, k)

        return indices, scores  # both are numpy arrays (B, k)

    def compute_topk_html(self, latent_array: np.ndarray, topk: int = 5, max_steps_display: int = 50) -> str:
        """
        Форматированный HTML вывод top-k для каждого временного шага.
        Показывает не более max_steps_display шагов (чтобы не засорять UI).
        """
        try:
            indices, scores = self.compute_topk(latent_array, topk=topk)
        except Exception as e:
            return f"<b>compute_topk error:</b> {e}"

        # В случае 1D входа indices shape (1,k)
        if indices.ndim == 1:
            indices = indices.reshape(1, -1)
            scores = scores.reshape(1, -1)

        lines = []
        total_steps = indices.shape[0]
        display_steps = min(total_steps, max_steps_display)
        for _t in range(total_steps):
            t = total_steps - _t - 1
            step_inds = indices[t]
            step_scores = scores[t]
            pairs = [f"{i}:{s:.4f}" for i, s in zip(step_inds, step_scores)]
            lines.append(f"t{t}: " + ", ".join(pairs))

        html = "<pre style='margin:0;'>" + "\n".join(lines) + "</pre>"
        return html

    def recalculate(self):
        """
        Применить edits к scenario, вызвать compute_by_scenario(scenario),
        загрузить новый scenario и обновить граф/данные.
        """
        # 1) Сохраняем старую форма (чтобы понять, нужно ли перестраивать структуру)
        old_positions = len(self.editable_chains) if self.editable_chains is not None else None

        # 2) Применяем правки в scenario
        self._apply_edits_to_scenario()

        # 3) Вызываем внешнюю функцию пересчёта — ожидается, что вернёт новый scenario
        #    (имя функции в вашем проекте compute_by_scenario)
        try:
            # если last_modified не установлен — передадим текущий уровень и позицию 0
            if self.last_modified is None:
                call_level = int(self.current_level)
                call_pos = 0
            else:
                call_level, call_pos = int(self.last_modified[0]), int(self.last_modified[1])

            new_scenario = self.hmoe.recompute_by_scenario(self.scenario, level=call_level, position=call_pos)
        except NameError:
            raise RuntimeError(
                "compute_by_scenario(scenario) не найден. Импортируйте функцию или определите её в области видимости.")
        except Exception as e:
            raise RuntimeError(f"compute_by_scenario вызвал ошибку: {e}")

        # 4) Подставляем новый сценарий и загружаем данные
        self.scenario = new_scenario
        self._load_from_scenario()

        # 5) Решаем, нужно ли перестраивать структуру (изменилось число позиций)
        new_positions = len(self.editable_chains)
        shape_changed = (old_positions is None) or (new_positions != old_positions)

        if shape_changed:
            # структура изменилась — безопасно пересоздать граф (create_graph() у вас создаёт новый widget)
            self.create_graph()
            return

        # 6) Иначе — обновляем данные в уже существующем графе (без пересоздания виджета)
        try:
            # Обновляем веса и подписи у рёбер
            for edge in self.cytoscape.graph.edges:
                data = edge['data']
                # рёбра с 'weight' — наши "весные" ребра
                if 'weight' in data and isinstance(data.get('source', ''), str) and data['source'].startswith('E'):
                    src = data['source']  # пример: "E3_1"
                    try:
                        parts = src[1:].split('_')
                        pos = int(parts[0])
                        k = int(parts[1])
                    except Exception:
                        continue
                    w = float(self.editable_weights[pos, k])
                    data['weight'] = w
                    # поле strval у вас используется для label
                    data['strval'] = f'{w:.3f}'

            # Обновляем узлы-суммы L{c}
            for node in self.cytoscape.graph.nodes:
                nid = node['data']['id']
                if isinstance(nid, str) and nid.startswith('L'):
                    # индекс c
                    try:
                        c = int(nid[1:])
                    except Exception:
                        continue
                    sum_val = 0.0
                    if c > 0:
                        for r in range(self.top_k):
                            expert = int(self.editable_chains[c - 1, r])
                            w = float(self.editable_weights[c - 1, r])
                            sum_val += w * self.expert_values.get(expert, 0.5)
                    node['data']['value'] = sum_val

                # Обновляем мета-информацию экспертов (вдруг поменялся expert id)
                if isinstance(nid, str) and nid.startswith('E'):
                    try:
                        parts = nid[1:].split('_')
                        pos = int(parts[0]);
                        k = int(parts[1])
                    except Exception:
                        continue
                    expert_id = int(self.editable_chains[pos, k])
                    node['data']['expert'] = expert_id
                    node['data']['label'] = f"E{expert_id}"
                    node['data']['weight'] = float(self.editable_weights[pos, k])
                    # value может быть оставлен как есть или обновлён из expert_values
                    node['data']['value'] = self.expert_values.get(expert_id, node['data'].get('value', 0.5))

            # 7) Никакого пересоздания layout — просто попытаться обновить визуал
            try:
                # Некоторым версиям ipycytoscape требуется redraw чтобы обновить label/edge label
                self.cytoscape.graph.redraw()
            except Exception:
                # если redraw вызывает побочное поведение — просто не делать ничего
                pass

        except Exception as e:
            # Если что-то пошло не так при in-place обновлении — fallback к полной перестройке
            self.create_graph()
            return

    def display(self):
        """Отображение интерфейса"""
        clear_output(wait=True)

        main_layout = widgets.VBox([
            widgets.HTML("""
                <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            color: white; border-radius: 8px; margin-bottom: 15px;">
                    <h2 style="margin: 0;">📊 HMoE Table Visualizer</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Только эксперты E и суммы L, с таблицей редактирования</p>
                </div>
            """),
            widgets.HBox([
                self.control_panel,
                self.edit_panel
            ]),
            self.graph_container,
            self.table_container,
            widgets.HTML("""
                <div style="margin-top: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px;">
                    <b>Структура:</b><br>
                    • Каждая колонка = позиция (Pos 0, Pos 1, ...)<br>
                    • Строки в колонке = top-k эксперты (k=0 сверху, k=1 снизу)<br>
                    • Синие круги = эксперты (E0, E1, ...)<br>
                    • Зеленые ромбы = суммы L0, L1, ...<br>
                    • Цвета ребер соответствуют k-индексам<br>
                    • <b>Редактирование:</b> двойной клик в таблице ниже
                </div>
            """)
        ])

        display(main_layout)


# Тестовые данные
def create_final_test_scenario():
    """Создание тестовых данных"""
    chains = [
        torch.tensor([[
            [1, 2],  # Pos 0: эксперты 1 и 2
            [0, 1],  # Pos 1: эксперты 0 и 1
            [2, 0],  # Pos 2: эксперты 2 и 0
            [1, 3]  # Pos 3: эксперты 1 и 3
        ]]),
    ]

    weights = [
        [torch.tensor([
            [0.6, 0.4],  # Веса для Pos 0
            [0.7, 0.3],  # Веса для Pos 1
            [0.5, 0.5],  # Веса для Pos 2
            [0.8, 0.2]  # Веса для Pos 3
        ])],
    ]

    latents = [
        [torch.randn(4, 64)],
    ]

    class Scenario:
        def __init__(self, chains, weights, latents):
            self.chains = chains
            self.weights = weights
            self.latents = latents
            self.last_modified = None

    return Scenario(chains, weights, latents)


# CSS стили
def add_final_styles():
    display(HTML("""
    <style>
        .final-visualizer {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .expert-node {
            transition: all 0.3s;
        }

        .expert-node:hover {
            transform: scale(1.1);
            box-shadow: 0 0 10px rgba(52, 152, 219, 0.5);
        }

        .sum-node {
            transition: all 0.3s;
        }

        .sum-node:hover {
            transform: scale(1.1);
            box-shadow: 0 0 10px rgba(46, 204, 113, 0.5);
        }

        .widget-button {
            margin: 2px;
        }

        h4 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }

        .cytoscape-widget {
            border-radius: 8px;
            overflow: hidden;
        }

        .edge-k0 {
            opacity: 0.8;
        }

        .edge-k1 {
            opacity: 0.8;
        }

        .edge-selected {
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }

        .widget-text input {
            border-radius: 4px;
            border: 1px solid #ddd;
            padding: 4px 8px;
            text-align: center;
        }

        .widget-text input:focus {
            border-color: #3498db;
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
            outline: none;
        }
    </style>
    """))

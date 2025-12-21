import json
import requests
import pandas as pd
import streamlit as st
from utils import smiles_to_svg, svg_to_datauri

if 'results_df' not in st.session_state:
    st.session_state.results_df = None

if 'request_error' not in st.session_state:
    st.session_state.request_error = None

if 'number_of_filters' not in st.session_state:
    st.session_state.number_of_filters = 0

st.set_page_config(layout = 'wide')
st.subheader('Property Prediction')
st.divider(width = 'stretch')

with st.container(border = False, key = 'layout_container', height = 'content'):
   
    with st.expander(label = 'Settings', icon = ':material/settings:', expanded = True, width = 'stretch'):
        
        with st.container(border = False, horizontal = True, key = 'top_bar', height = 185):
            col1_1, col1_2, col1_3, col1_4 = st.columns([3, 2, 4, 1], gap = 'small')
            
            with col1_1:
                with st.container(border = True, 
                                key = 'container_1'):
                    uploaded_file = st.file_uploader('Choose a CSV file', type=['csv'])
            
            with col1_2:
                with st.container(border = True, 
                                    key = 'container_2',
                                    gap = 'small'):
                    if not uploaded_file:
                        st.selectbox(label = 'Select ID column', options = [''], disabled = True) 
                        st.selectbox(label = 'Select SMILES column', options = [''], disabled = True)
                        smiles_column = None
                        id_column = None
                    else:
                        molecules_df = pd.read_csv(uploaded_file)
                        id_column = st.selectbox(
                            label = 'Select ID column',
                            options = molecules_df.columns,
                            index = None
                        )
                        smiles_column = st.selectbox(
                            label = 'Select SMILES column',
                            options = molecules_df.columns,
                            index = None
                        )
            
            with col1_3:
                with st.container(border = True, key = 'container_3', horizontal = False, height = 'stretch', vertical_alignment = 'distribute'):
                    if not uploaded_file or not smiles_column or not id_column:
                        st.multiselect('Select models', options = [''], disabled = True)
                        st.popover('Show models specs', width = 'stretch', disabled = True)
                    else:
                        with open('./config/model_configs.json') as f:
                            model_configs = json.load(f)
                        model_names = [c['name'] for c in model_configs]
                        chosen_models = st.multiselect(
                            "Select models",
                            model_names,
                            default = model_names,
                            key = 'multiselect_models'
                            )
                        specs_df = pd.DataFrame(model_configs)[['name', 'algorithm', 'r squared']]
                        specs_df = (
                            specs_df.style
                                .applymap(lambda v: 'color: red;' if v < 0.4 else 'color: green;', subset=['r squared'])
                                .format({'r squared':'{:,.2f}'})
                        )
                        with st.popover('Show models specs', width = 'stretch'):
                            st.write('Placeholder with made up numbers')
                            st.dataframe(specs_df, hide_index = True)
            
            with col1_4:
                with st.container(border = False, height = 'stretch', vertical_alignment = 'center', key = 'button_container'):
                    if not uploaded_file or not smiles_column or not id_column or not chosen_models:
                        st.button('Predict', icon = ':material/play_circle:', width = 'stretch', disabled = True, key = 'predict_button_disabled')   
                    else:
                        if st.button('Predict', icon = ':material/play_circle:', use_container_width = True, type = 'primary', key = 'predict_button_enabled'):
                            payload_rows_df = molecules_df[[id_column, smiles_column]].rename(columns = {id_column: 'id', smiles_column: 'smiles'})
                            payload = {'config': {'models': chosen_models}, 'data': {'rows': payload_rows_df.to_dict(orient = 'records')}}
                            try:
                                r = requests.post('http://127.0.0.1:8000/predict', json = payload)
                                r.raise_for_status()
                                st.session_state.results_df = pd.DataFrame(r.json())
                                st.session_state.results_df.rename(columns = {'id': id_column, 'smiles': smiles_column}, inplace = True)
                                st.session_state.request_error = None
                            except requests.exceptions.RequestException as e:
                                st.session_state.results_df = None
                                st.session_state.request_error = f'Request failed: {e}'

    if st.session_state.results_df is None and st.session_state.request_error is None: 
        with st.container(border = True, height = 780, vertical_alignment = 'center', horizontal_alignment = 'center', key = 'results_before_predict'):
            st.markdown(
                "<div style='text-align: center;'>Load compounds and define settings to start predictions.</div>",
                unsafe_allow_html = True,
                )
    elif st.session_state.request_error is not None:
        st.error(st.session_state.request_error)
    else:
        with st.container(border = True, height = 780, key = 'results_after_predict'):
            results_df = st.session_state.results_df.copy()
            results_df['svg_text'] = results_df[smiles_column].apply(smiles_to_svg)
            # insert rendering column after ID column (second from left)
            results_df.insert(
                        1,
                        'Structure_SVG',
                        value = results_df['svg_text'].apply(svg_to_datauri),
                        allow_duplicates = True
                        )
            # merge predictions with molecules_df
            results_df = results_df.merge(molecules_df.drop(columns = [smiles_column]), on = id_column)
            # move SMILES column to end of df
            results_df[smiles_column] = results_df.pop(smiles_column)
            results_df.drop(columns = ['svg_text'], inplace = True)
            total_rows = results_df.shape[0]
            with st.container(border = False, width = 'stretch', key = 'selection_option_container'):
                selection_option = st.segmented_control(
                                                    label = 'Selection options', 
                                                    options = ['Select all', 'Deselect all'],
                                                    label_visibility = 'collapsed',
                                                    default = 'Select all'
                                                    )
                
                if selection_option == 'Select all':
                    col_selection_bool = True
                elif selection_option == 'Deselect all':
                    col_selection_bool = False
            
            col2_1, col2_2 = st.columns([3, 1], gap = 'small')
            
            with col2_2:
                with st.expander('Search by ID', icon = ':material/id_card:', width = 'stretch', expanded = False):
                    id_string = st.text_input('Search one or more IDs, separated by comma', label_visibility = 'visible')
                    if id_string:
                        id_list = [e.strip() for e in id_string.split(',') if e.strip()]
                        results_df = results_df[results_df[id_column].isin(id_list)]
                with st.expander('Numerical filters', icon = ':material/filter_alt:', width = 'stretch', expanded = False):
                    numeric_columns = (
                                    results_df
                                    .select_dtypes(include = 'number')
                                    .loc[:, lambda df: df.nunique() > 1]   # cols need > 1 value for valid slider range
                                    .columns
                                    )

                    with st.container(border = False, horizontal = True):
                        if st.button('\-', width = 'stretch', type = 'tertiary'):
                            st.session_state.number_of_filters -= 1
                        if st.button('\+', width = 'stretch', type = 'tertiary'):
                            st.session_state.number_of_filters += 1

                        max_filters = 4
                        st.session_state.number_of_filters = min(max(st.session_state.number_of_filters, 0), max_filters)

                    with st.container(border = False, height = 'stretch', key = 'filters_container', gap = 'small', vertical_alignment = 'distribute'):        
                        for i in range(st.session_state.number_of_filters):
                            with st.container(border = True, horizontal = True, height = 76, vertical_alignment = 'center', key = f'filter_{i}'):
                                column_to_filter = st.selectbox('Select column to filter', 
                                                                options = numeric_columns,
                                                                label_visibility = 'collapsed', 
                                                                width = 'stretch',
                                                                key = f'col_select_{i}')
                                min_value = results_df[column_to_filter].min()
                                max_value = results_df[column_to_filter].max()
                                value_range = st.slider(f'{column_to_filter} slider', 
                                                        min_value = min_value, 
                                                        max_value = max_value, 
                                                        value = (min_value, max_value), 
                                                        label_visibility = 'collapsed',
                                                        width = 'stretch',
                                                        key = f'filter_slider_{i}')
                                results_df = results_df[results_df[column_to_filter].between(*value_range)]
            
            with col2_1:
                results_df.insert(
                                0,
                                'Selected',
                                value = col_selection_bool,
                                allow_duplicates = True
                                )
                edited_df = st.data_editor(
                                        results_df, 
                                        column_config = {
                                                    'Structure_SVG': st.column_config.ImageColumn(width = 'large'), 
                                                    'Selected': st.column_config.CheckboxColumn(default = False, disabled = False)
                                                    },
                                        row_height = 120,
                                        height = 640, 
                                        hide_index = True,
                                        disabled = results_df.columns[1:],
                                        key = 'results_data_editor'
                                        )
                selected_df = edited_df[edited_df['Selected'] == True]
                download_df = selected_df.drop(columns = ['Selected', 'Structure_SVG'])
                smiles_column_to_move = download_df.pop(smiles_column)
                download_df.insert(1, smiles_column, smiles_column_to_move)
                csv_file = download_df.to_csv(index = False)

            with st.container(border = False, horizontal = True, height = 'stretch', vertical_alignment = 'center'):
                st.caption(f'Total {total_rows} | Visible {results_df.shape[0]} | Selected {selected_df.shape[0]}')
                st.download_button(
                                label = 'Download selected', 
                                data = csv_file, 
                                file_name = f'{uploaded_file.name}_predictions.csv', 
                                type = 'primary', 
                                icon = ':material/download:'
                                )


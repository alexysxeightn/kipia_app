import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import plotly.express as px
import io

# Настройка страницы
st.set_page_config(page_title="Комплексная система учета КИПиА", page_icon="📊🤖", layout="wide")


# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('kipia_combined.db')
    cursor = conn.cursor()

    # Таблица приборов (расширенная версия из v2)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS instruments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        factory_number TEXT UNIQUE,
        measurement_range TEXT,
        accuracy TEXT,
        installation_location TEXT NOT NULL,
        last_verification_date TEXT,
        next_verification_date TEXT,
        status TEXT NOT NULL,
        notes TEXT,
        criticality INTEGER DEFAULT 1
    )
    ''')

    # Таблица отказов (расширенная версия из v2)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS failures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        instrument_id INTEGER NOT NULL,
        failure_date TEXT NOT NULL,
        description TEXT NOT NULL,
        reason TEXT,
        actions_taken TEXT,
        repair_date TEXT,
        repair_status TEXT NOT NULL,
        responsible_person TEXT,
        used_parts TEXT,
        FOREIGN KEY (instrument_id) REFERENCES instruments (id)
    );
    ''')

    # Таблица склада (расширенная версия из v2)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS inventory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        part_number TEXT UNIQUE,
        quantity INTEGER NOT NULL,
        min_quantity INTEGER,
        optimal_quantity INTEGER,
        unit TEXT DEFAULT 'шт.',
        location TEXT,
        supplier TEXT,
        lead_time INTEGER DEFAULT 14,
        notes TEXT,
        last_update TEXT,
        failure_rate FLOAT DEFAULT 0
    )
    ''')

    # Таблица связей приборов и запчастей (новая из v2)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS instrument_parts (
        instrument_id INTEGER NOT NULL,
        part_id INTEGER NOT NULL,
        usage_count INTEGER DEFAULT 1,
        PRIMARY KEY (instrument_id, part_id),
        FOREIGN KEY (instrument_id) REFERENCES instruments (id),
        FOREIGN KEY (part_id) REFERENCES inventory (id)
    )
    ''')

    conn.commit()
    return conn


conn = init_db()


# Общие функции для работы с данными
def get_data(query, params=None):
    return pd.read_sql(query, conn, params=params)


def execute_query(query, params=None):
    cursor = conn.cursor()
    cursor.execute(query, params or ())
    conn.commit()
    return cursor.lastrowid


# Функции из v1 (адаптированные под новую схему)
def add_instrument(data):
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO instruments (
        name, type, factory_number, measurement_range, accuracy,
        installation_location, last_verification_date, next_verification_date, status, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', data)
    conn.commit()


def update_instrument(instrument_id, data):
    cursor = conn.cursor()
    cursor.execute('''
    UPDATE instruments SET
        name=?, type=?, factory_number=?, measurement_range=?, accuracy=?,
        installation_location=?, last_verification_date=?, next_verification_date=?, status=?, notes=?
    WHERE id=?
    ''', (*data, instrument_id))
    conn.commit()


def delete_instrument(instrument_id):
    cursor = conn.cursor()
    cursor.execute('DELETE FROM instruments WHERE id=?', (instrument_id,))
    conn.commit()


def get_all_instruments():
    return get_data('SELECT * FROM instruments ORDER BY next_verification_date')


def get_instrument_by_id(instrument_id):
    return get_data('SELECT * FROM instruments WHERE id=?', (instrument_id,)).iloc[0]


def add_failure(data):
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO failures (
        instrument_id, failure_date, description, reason, actions_taken,
        repair_date, repair_status, responsible_person
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', data)
    conn.commit()


def get_all_failures():
    return get_data('''
    SELECT f.*, i.name as instrument_name, i.factory_number 
    FROM failures f
    LEFT JOIN instruments i ON f.instrument_id = i.id
    ORDER BY f.failure_date DESC
    ''')


def add_inventory_item(data):
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO inventory (
        name, type, part_number, quantity, min_quantity,
        unit, location, supplier, notes, last_update
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', data)
    conn.commit()


def get_all_inventory():
    return get_data('SELECT * FROM inventory ORDER BY name')


def get_low_stock_items():
    return get_data('''
    SELECT * FROM inventory 
    WHERE min_quantity IS NOT NULL AND quantity <= min_quantity
    ORDER BY quantity
    ''')


# Функции машинного обучения из v2
def prepare_failure_data():
    ten_years_ago = (datetime.now() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')
    query = '''
    SELECT 
        i.type as instrument_type,
        ip.part_id,
        strftime('%Y', f.failure_date) as year,
        COUNT(*) as failures_count
    FROM failures f
    JOIN instruments i ON f.instrument_id = i.id
    JOIN instrument_parts ip ON i.id = ip.instrument_id
    WHERE f.failure_date >= ?
    GROUP BY i.type, ip.part_id, strftime('%Y', f.failure_date)
    '''
    failure_data = get_data(query, (ten_years_ago,))

    if failure_data.empty:
        return None

    pivot_data = failure_data.pivot_table(
        index=['instrument_type', 'part_id'],
        columns='year',
        values='failures_count',
        fill_value=0
    ).reset_index()

    pivot_data['total_failures'] = pivot_data.iloc[:, 2:].sum(axis=1)
    return pivot_data


def train_failure_model():
    data = prepare_failure_data()
    if data is None or len(data) < 2:
        return None

    X = data[['total_failures']]
    y = data.iloc[:, 2:-1].mean(axis=1)
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X, y)
    return model


def predict_parts_demand():
    model = train_failure_model()
    if model is None:
        return None

    parts_query = '''
    SELECT 
        i.id as part_id,
        i.name,
        i.type,
        i.quantity,
        i.min_quantity,
        i.optimal_quantity,
        COALESCE(SUM(ip.usage_count), 0) as total_usage,
        COALESCE(COUNT(DISTINCT ip.instrument_id), 0) as instruments_count
    FROM inventory i
    LEFT JOIN instrument_parts ip ON i.id = ip.part_id
    GROUP BY i.id
    '''
    parts_data = get_data(parts_query)

    if parts_data.empty:
        return None

    failure_data = prepare_failure_data()
    if failure_data is not None:
        parts_data = parts_data.merge(
            failure_data[['part_id', 'total_failures']],
            on='part_id',
            how='left'
        )
        parts_data['total_failures'] = parts_data['total_failures'].fillna(0)
    else:
        parts_data['total_failures'] = 0

    X_pred = parts_data[['total_failures']]
    parts_data['predicted_demand'] = model.predict(X_pred).clip(0)
    parts_data['recommended_quantity'] = (parts_data['predicted_demand'] * 1.2).round().astype(int)
    parts_data['is_low_stock'] = (
            (parts_data['quantity'] < parts_data['recommended_quantity']) |
            ((parts_data['min_quantity'] > 0) & (parts_data['quantity'] < parts_data['min_quantity'])
             ))

    return parts_data.sort_values(by='is_low_stock', ascending=False)


# Функции для импорта из Excel
def import_from_excel(uploaded_file, table_name):
    try:
        df = pd.read_excel(uploaded_file)

        if table_name == "instruments":
            # Преобразование дат в строки
            date_columns = ['last_verification_date', 'next_verification_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')

            # Заполнение обязательных полей
            if 'status' not in df.columns:
                df['status'] = 'В работе'
            if 'criticality' not in df.columns:
                df['criticality'] = 1

        elif table_name == "failures":
            if 'failure_date' in df.columns:
                df['failure_date'] = pd.to_datetime(df['failure_date']).dt.strftime('%Y-%m-%d')
            if 'repair_date' in df.columns:
                df['repair_date'] = pd.to_datetime(df['repair_date']).dt.strftime('%Y-%m-%d')
            if 'repair_status' not in df.columns:
                df['repair_status'] = 'Не начат'

        elif table_name == "inventory":
            if 'unit' not in df.columns:
                df['unit'] = 'шт.'
            if 'lead_time' not in df.columns:
                df['lead_time'] = 14
            if 'failure_rate' not in df.columns:
                df['failure_rate'] = 0
            if 'last_update' not in df.columns:
                df['last_update'] = datetime.now().strftime('%Y-%m-%d')

        # Запись в базу данных
        df.to_sql(table_name, conn, if_exists='append', index=False)
        return True, f"Данные успешно импортированы в таблицу {table_name}"
    except Exception as e:
        return False, f"Ошибка при импорте: {str(e)}"


# Интерфейс Streamlit
st.title("📊🤖 Комплексная система учета КИПиА с аналитикой")

# Главное меню
menu_options = [
    "Классический интерфейс",
    "Интерфейс с аналитикой",
    "Импорт данных из Excel"
]
main_choice = st.sidebar.selectbox("Режим работы", menu_options)

if main_choice == "Классический интерфейс":
    st.header("Классический интерфейс учета КИПиА")

    # Меню действий (из v1)
    action_options = [
        "Просмотр всех приборов",
        "Добавить прибор",
        "Редактировать прибор",
        "Удалить прибор",
        "Журнал отказов",
        "Добавить запись об отказе",
        "Складские запасы",
        "Добавить складскую позицию",
        "Просмотр дефицитных позиций"
    ]
    action = st.sidebar.selectbox("Выберите действие", action_options)

    if action == "Просмотр всех приборов":
        st.subheader("Список всех приборов")
        df = get_all_instruments()

        if not df.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_type = st.selectbox("Фильтр по типу", ["Все"] + list(df['type'].unique()))
            with col2:
                filter_location = st.selectbox("Фильтр по месту установки",
                                               ["Все"] + list(df['installation_location'].unique()))
            with col3:
                filter_status = st.selectbox("Фильтр по статусу", ["Все"] + list(df['status'].unique()))

            if filter_type != "Все":
                df = df[df['type'] == filter_type]
            if filter_location != "Все":
                df = df[df['installation_location'] == filter_location]
            if filter_status != "Все":
                df = df[df['status'] == filter_status]

            st.dataframe(df, use_container_width=True)

            st.download_button(
                label="Экспорт в CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='kipia_instruments.csv',
                mime='text/csv'
            )
        else:
            st.warning("База данных приборов пуста")

    elif action == "Добавить прибор":
        st.subheader("Добавление нового прибора")
        with st.form("add_form"):
            col1, col2 = st.columns(2)

            with col1:
                name = st.text_input("Наименование прибора*", max_chars=100)
                type_ = st.text_input("Тип прибора*", max_chars=50)
                factory_number = st.text_input("Заводской номер*", max_chars=50)
                measurement_range = st.text_input("Диапазон измерений", max_chars=50)
                accuracy = st.text_input("Класс точности", max_chars=20)

            with col2:
                installation_location = st.text_input("Место установки*", max_chars=100)
                last_verification_date = st.date_input("Дата последней поверки")
                next_verification_date = st.date_input("Дата следующей поверки")
                status = st.selectbox("Статус*", ["В работе", "На поверке", "В ремонте", "Списан"])
                notes = st.text_area("Примечания", max_chars=500)

            submitted = st.form_submit_button("Добавить прибор")

            if submitted:
                if not name or not type_ or not factory_number or not installation_location:
                    st.error("Поля с * обязательны для заполнения")
                else:
                    data = (
                        name, type_, factory_number, measurement_range, accuracy,
                        installation_location,
                        last_verification_date.isoformat() if last_verification_date else None,
                        next_verification_date.isoformat() if next_verification_date else None,
                        status, notes
                    )
                    add_instrument(data)
                    st.success("Прибор успешно добавлен!")

    elif action == "Журнал отказов":
        st.subheader("Журнал отказов КИПиА")
        df = get_all_failures()

        if not df.empty:
            col1, col2 = st.columns(2)
            with col1:
                filter_status = st.selectbox("Фильтр по статусу ремонта", ["Все"] + list(df['repair_status'].unique()))
            with col2:
                filter_instrument = st.selectbox("Фильтр по прибору", ["Все"] + list(df['instrument_name'].unique()))

            if filter_status != "Все":
                df = df[df['repair_status'] == filter_status]
            if filter_instrument != "Все":
                df = df[df['instrument_name'] == filter_instrument]

            st.dataframe(df, use_container_width=True)

            st.download_button(
                label="Экспорт журнала отказов",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='failures_log.csv',
                mime='text/csv'
            )
        else:
            st.warning("Журнал отказов пуст")

    elif action == "Складские запасы":
        st.subheader("Учет складских запасов")
        df = get_all_inventory()

        if not df.empty:
            col1, col2 = st.columns(2)
            with col1:
                filter_type = st.selectbox("Фильтр по типу", ["Все"] + list(df['type'].unique()))
            with col2:
                filter_location = st.selectbox("Фильтр по месту хранения", ["Все"] + list(df['location'].unique()))

            if filter_type != "Все":
                df = df[df['type'] == filter_type]
            if filter_location != "Все":
                df = df[df['location'] == filter_location]

            st.dataframe(df, use_container_width=True)

            st.download_button(
                label="Экспорт данных о запасах",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='inventory.csv',
                mime='text/csv'
            )
        else:
            st.warning("Складские запасы не добавлены")

    # Остальные действия из v1 можно добавить аналогично

elif main_choice == "Интерфейс с аналитикой":
    st.header("Интерфейс с аналитикой и прогнозированием")

    # Меню аналитики (из v2)
    analytics_options = [
        "Обзор приборов",
        "Журнал отказов",
        "Складские запасы",
        "Прогнозирование дефицита",
        "Аналитика"
    ]
    analytics_choice = st.sidebar.selectbox("Меню аналитики", analytics_options)

    if analytics_choice == "Обзор приборов":
        st.subheader("Список приборов")
        instruments = get_data("SELECT * FROM instruments")
        st.dataframe(instruments, use_container_width=True)

    elif analytics_choice == "Журнал отказов":
        st.subheader("Журнал отказов")
        failures = get_data('''
        SELECT f.*, i.name as instrument_name 
        FROM failures f JOIN instruments i ON f.instrument_id = i.id
        ORDER BY f.failure_date DESC
        ''')
        st.dataframe(failures, use_container_width=True)

    elif analytics_choice == "Складские запасы":
        st.subheader("Складские запасы")
        inventory = get_data("SELECT * FROM inventory ORDER BY quantity ASC")
        st.dataframe(inventory, use_container_width=True)

    elif analytics_choice == "Прогнозирование дефицита":
        st.subheader("Прогнозирование дефицита запчастей")

        with st.spinner("Анализируем данные..."):
            prediction = predict_parts_demand()

        if prediction is None:
            st.warning("Недостаточно данных для прогнозирования, нужно больше данных об отказах.")
        else:
            st.subheader("Рекомендуемые уровни запасов")

            fig = px.bar(
                prediction,
                x='name',
                y=['quantity', 'recommended_quantity'],
                barmode='group',
                labels={'value': 'Количество', 'name': 'Запчасть'},
                color_discrete_map={
                    'quantity': 'blue',
                    'recommended_quantity': 'red'
                }
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                prediction[['name', 'type', 'quantity', 'min_quantity',
                            'recommended_quantity', 'is_low_stock']],
                use_container_width=True
            )

            if st.button("Обновить оптимальные запасы на складе"):
                for _, row in prediction.iterrows():
                    execute_query(
                        "UPDATE inventory SET optimal_quantity=? WHERE id=?",
                        (row['recommended_quantity'], row['part_id'])
                    )
                st.success("Оптимальные запасы обновлены!")

    elif analytics_choice == "Аналитика":
        st.subheader("Аналитика отказов и запасов")

        st.subheader("Частота отказов по типам приборов")
        failure_by_type = get_data('''
        SELECT i.type, COUNT(*) as failures_count
        FROM failures f JOIN instruments i ON f.instrument_id = i.id
        GROUP BY i.type
        ORDER BY failures_count DESC
        ''')

        if not failure_by_type.empty:
            fig1 = px.pie(
                failure_by_type,
                names='type',
                values='failures_count',
                title='Распределение отказов по типам приборов'
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("Нет данных об отказах для анализа")

        st.subheader("Топ запчастей по расходу")
        parts_usage = get_data('''
        SELECT i.name, i.type, SUM(ip.usage_count) as total_usage
        FROM instrument_parts ip JOIN inventory i ON ip.part_id = i.id
        GROUP BY i.id
        ORDER BY total_usage DESC
        LIMIT 10
        ''')

        if not parts_usage.empty:
            fig2 = px.bar(
                parts_usage,
                x='name',
                y='total_usage',
                color='type',
                labels={'name': 'Запчасть', 'total_usage': 'Количество использований'}
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Нет данных о расходе запчастей")

elif main_choice == "Импорт данных из Excel":
    st.header("Импорт данных из Excel")
    table_options = {
        "Приборы": "instruments",
        "Отказы": "failures",
        "Складские запасы": "inventory",
        "Связи приборов и запчастей": "instrument_parts"
    }
    selected_table = st.selectbox("Выберите таблицу для импорта", list(table_options.keys()))
    table_name = table_options[selected_table]
    
    uploaded_file = st.file_uploader(f"Загрузите Excel-файл для таблицы {selected_table}", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        st.info("Перед импортом убедитесь, что структура файла соответствует таблице:")
        
        # Показываем пример структуры таблицы
        if table_name == "instruments":
            st.write("Ожидаемые колонки: name, type, factory_number, measurement_range, accuracy, installation_location, last_verification_date, next_verification_date, status, notes")
        elif table_name == "failures":
            st.write("Ожидаемые колонки: instrument_id, failure_date, description, reason, actions_taken, repair_date, repair_status, responsible_person")
        elif table_name == "inventory":
            st.write("Ожидаемые колонки: name, type, part_number, quantity, min_quantity, unit, location, supplier, notes")
        elif table_name == "instrument_parts":
            st.write("Ожидаемые колонки: instrument_id, part_id, usage_count")

        if st.button("Начать импорт"):
            try:
                df = pd.read_excel(uploaded_file)
                
                # Проверка наличия обязательных колонок для instrument_parts
                if table_name == "instrument_parts":
                    required_columns = ['instrument_id', 'part_id', 'usage_count']
                    if not all(col in df.columns for col in required_columns):
                        st.error("Файл не содержит всех необходимых колонок: instrument_id, part_id, usage_count")
                        st.stop()
                    else:
                        df['usage_count'] = pd.to_numeric(df['usage_count'], errors='coerce').fillna(1).astype(int)

                # Запись в базу данных
                df.to_sql(table_name, conn, if_exists='append', index=False)
                st.success(f"✅ Данные успешно импортированы в таблицу `{table_name}`")
                
            except Exception as e:
                st.error(f"❌ Ошибка при импорте: {str(e)}")

# Закрытие соединения с базой данных при завершении работы
conn.close()
// Основной класс для управления весами ошибок
class LossWeightsManager {
    constructor() {
        this.tableBody = document.getElementById('lossWeightsBody');
        this.hiddenInput = document.getElementById('loss_weights_config');
        this.schedulePreview = document.getElementById('schedulePreview');
        this.epochWarning = document.getElementById('epochWarning');
        this.totalEpochsInput = document.getElementById('num_epoch');

        this.init();
    }

    init() {
        this.addEventListeners();
        this.updateHiddenField();

        // Добавляем начальный пример при первом посещении
        if (this.tableBody && this.tableBody.children.length === 0) {
            this.addExampleConfig();
        }
    }

    addEventListeners() {
        // Кнопки управления
        const addButton = document.getElementById('addWeightRow');
        const resetButton = document.getElementById('resetWeights');

        if (addButton) addButton.addEventListener('click', () => this.addRow());
        if (resetButton) resetButton.addEventListener('click', () => this.resetToDefault());

        // Следим за изменением общего количества эпох
        if (this.totalEpochsInput) {
            this.totalEpochsInput.addEventListener('input', () => this.updateHiddenField());
        }

        // Делегирование событий для динамически добавленных элементов
        if (this.tableBody) {
            this.tableBody.addEventListener('input', (e) => {
                if (e.target.classList.contains('weight-input') || e.target.classList.contains('epoch-input')) {
                    this.updateHiddenField();
                }
            });

            this.tableBody.addEventListener('click', (e) => {
                if (e.target.classList.contains('btn-remove-row')) {
                    e.target.closest('tr').remove();
                    this.updateHiddenField();
                }
            });
        }
    }

    addExampleConfig() {
        // Добавляем стандартные значения которые видны в форме
        this.addRow(0, '10.0', '1.0');
        this.addRow(1000, '3.0', '1.0');
        this.addRow(5000, '1.0', '1.0');
    }

    addRow(epoch = '', dataWeight = '1.0', pdeWeight = '1.0') {
        if (!this.tableBody) return;

        const row = document.createElement('tr');
        row.innerHTML = `
            <td>
                <input type="number" class="form-control form-control-sm epoch-input"
                       min="0" value="${epoch}" placeholder="Эпоха">
            </td>
            <td>
                <input type="number" class="form-control form-control-sm weight-input"
                       step="0.1" min="0.1" value="${dataWeight}" placeholder="1.0">
            </td>
            <td>
                <input type="number" class="form-control form-control-sm weight-input"
                       step="0.1" min="0.1" value="${pdeWeight}" placeholder="1.0">
            </td>
            <td>
                <button type="button" class="btn btn-outline-danger btn-sm btn-remove-row" title="Удалить правило">
                    Удалить
                </button>
            </td>
        `;
        this.tableBody.appendChild(row);
        this.updateHiddenField();
    }

    resetToDefault() {
        if (this.tableBody) {
            this.tableBody.innerHTML = '';
        }
        this.updateHiddenField();
    }

    getSchedule() {
        if (!this.tableBody) return [];

        const rows = Array.from(this.tableBody.querySelectorAll('tr'));
        const schedule = [];

        for (const row of rows) {
            const epochInput = row.querySelector('.epoch-input');
            const dataWeightInput = row.querySelector('.weight-input:nth-child(1)');
            const pdeWeightInput = row.querySelector('.weight-input:nth-child(2)');

            if (epochInput && dataWeightInput && pdeWeightInput) {
                const epoch = parseInt(epochInput.value);
                const dataWeight = parseFloat(dataWeightInput.value);
                const pdeWeight = parseFloat(pdeWeightInput.value);

                if (!isNaN(epoch) && !isNaN(dataWeight) && !isNaN(pdeWeight)) {
                    schedule.push({ epoch, dataWeight, pdeWeight });
                }
            }
        }

        // Сортируем по эпохе
        schedule.sort((a, b) => a.epoch - b.epoch);
        return schedule;
    }

    updateHiddenField() {
        const schedule = this.getSchedule();
        const totalEpochs = this.totalEpochsInput ? parseInt(this.totalEpochsInput.value) || 0 : 0;

        // Проверяем, есть ли правила с эпохами больше общего количества
        const hasExceedingEpochs = schedule.some(item => item.epoch > totalEpochs);
        if (this.epochWarning) {
            this.epochWarning.style.display = hasExceedingEpochs ? 'block' : 'none';
        }

        // ВАЖНО: Даже если расписание пустое, мы должны передавать пустую строку,
        // чтобы сервер знал, что используются веса по умолчанию
        let configString = '';
        if (schedule.length > 0) {
            // Формируем строку конфигурации
            configString = schedule.map(item =>
                `${item.epoch}:${item.dataWeight},${item.pdeWeight}`
            ).join('; ');
        }

        if (this.hiddenInput) {
            this.hiddenInput.value = configString;
            console.log('Конфигурация весов обновлена:', configString);
        }

        // Обновляем предпросмотр
        this.updateSchedulePreview(schedule, totalEpochs);
    }

    updateSchedulePreview(schedule, totalEpochs) {
        if (!this.schedulePreview) return;

        if (schedule.length === 0) {
            this.schedulePreview.textContent = 'Равные веса [1.0, 1.0] на всех эпохах';
            return;
        }

        let preview = '';
        for (let i = 0; i < schedule.length; i++) {
            const current = schedule[i];
            const next = schedule[i + 1];

            // Проверяем, превышает ли текущая эпоха общее количество
            const epochExceeds = current.epoch > totalEpochs;

            if (i === 0 && current.epoch > 0) {
                preview += `<span>0-${current.epoch-1}: [1.0, 1.0]</span><br>`;
            }

            const rangeEnd = next ? next.epoch - 1 : totalEpochs;
            const displayRangeEnd = next ? Math.min(next.epoch - 1, totalEpochs) : totalEpochs;

            // Если текущая эпоха превышает общее количество, показываем предупреждение
            if (epochExceeds) {
                preview += `<span class="warning-text">${current.epoch}-${displayRangeEnd}: [${current.dataWeight}, ${current.pdeWeight}] (правило не будет применено)</span><br>`;
            } else {
                preview += `<span>${current.epoch}-${displayRangeEnd}: [${current.dataWeight}, ${current.pdeWeight}]</span><br>`;
            }
        }

        this.schedulePreview.innerHTML = preview;
    }

    loadConfig(configString) {
        this.resetToDefault();

        if (!configString) return;

        const entries = configString.split(';');
        entries.forEach(entry => {
            const [epochPart, weightsPart] = entry.split(':');
            if (epochPart && weightsPart) {
                const epoch = parseInt(epochPart.trim());
                const weights = weightsPart.split(',');
                if (weights.length === 2) {
                    const dataWeight = parseFloat(weights[0].trim());
                    const pdeWeight = parseFloat(weights[1].trim());
                    if (!isNaN(epoch) && !isNaN(dataWeight) && !isNaN(pdeWeight)) {
                        this.addRow(epoch, dataWeight, pdeWeight);
                    }
                }
            }
        });
    }
}

// Функции для управления формами
class FormManager {
    static init() {
        this.setupUploadForm();
        this.setupUpdateForm();
        this.setupMLForm();
        this.setupValidation();

        // Добавляем отладку для формы ML
        this.debugMLForm();
    }

    static setupUploadForm() {
        const form = document.getElementById('uploadForm');
        if (form) {
            form.addEventListener('submit', function() {
                const button = document.getElementById('uploadButton');
                const spinner = document.getElementById('uploadLoading');
                FormManager.showLoading(button, spinner, 'Загрузка...');
            });
        }
    }

    static setupUpdateForm() {
        const form = document.getElementById('updateCollocationForm');
        if (form) {
            form.addEventListener('submit', function() {
                const button = document.getElementById('updateCollocationButton');
                const spinner = document.getElementById('updateLoading');
                FormManager.showLoading(button, spinner, 'Обновление...');
            });
        }
    }

    static setupMLForm() {
    const form = document.getElementById('mlForm');
    if (form) {
        form.addEventListener('submit', function(e) {
            console.log('=== ДЕТАЛЬНАЯ ПРОВЕРКА ФОРМЫ ===');

            // Принудительно обновляем скрытое поле
            if (window.weightsManager) {
                window.weightsManager.updateHiddenField();
            }

            const hiddenField = document.getElementById('loss_weights_config');
            console.log('Скрытое поле value:', hiddenField.value);
            console.log('Скрытое поле длина:', hiddenField.value.length);

            // Проверяем есть ли строки в таблице
            const tableRows = document.querySelectorAll('#lossWeightsBody tr');
            console.log('Количество строк в таблице:', tableRows.length);

            tableRows.forEach((row, index) => {
                const epochInput = row.querySelector('.epoch-input');
                const dataWeightInput = row.querySelector('.weight-input:nth-child(1)');
                const pdeWeightInput = row.querySelector('.weight-input:nth-child(2)');

                console.log(`Строка ${index}:`, {
                    epoch: epochInput?.value,
                    dataWeight: dataWeightInput?.value,
                    pdeWeight: pdeWeightInput?.value
                });
            });

            // Проверяем FormData
            const formData = new FormData(form);
            console.log('FormData содержимое:');
            for (let [key, value] of formData.entries()) {
                console.log(`  ${key}: ${value} (длина: ${value.length})`);
            }

            const button = document.getElementById('mlButton');
            const spinner = document.getElementById('mlLoading');
            FormManager.showLoading(button, spinner, 'Выполняется...');
        });
    }
}

    static debugMLForm() {
        // Добавляем отладочную информацию для формы ML
        const form = document.getElementById('mlForm');
        if (form) {
            form.addEventListener('click', function(e) {
                if (e.target.type === 'submit') {
                    console.log('Кнопка ML нажата, текущая конфигурация весов:',
                                document.getElementById('loss_weights_config')?.value);
                }
            });
        }
    }

    static setupValidation() {
        // Валидация параметров ML
        const numLayers = document.getElementById('num_layers');
        const numPerceptrons = document.getElementById('num_perceptrons');
        const numEpoch = document.getElementById('num_epoch');

        if (numLayers) {
            numLayers.addEventListener('change', function() {
                const value = parseInt(this.value);
                if (value < 1) this.value = 1;
                if (value > 50) this.value = 50;
            });
        }

        if (numPerceptrons) {
            numPerceptrons.addEventListener('change', function() {
                const value = parseInt(this.value);
                if (value < 10) this.value = 10;
                if (value > 200) this.value = 200;
            });
        }

        if (numEpoch) {
            numEpoch.addEventListener('change', function() {
                const value = parseInt(this.value);
                if (value < 100) this.value = 100;
                if (value > 100000) this.value = 100000;
                // Округление до ближайшей 1000
                this.value = Math.round(value / 1000) * 1000;
            });
        }
    }

    static validateEpochs() {
        const totalEpochs = parseInt(document.getElementById('num_epoch')?.value) || 0;
        const schedule = Array.from(document.querySelectorAll('.epoch-input'));

        let hasInvalidEpochs = false;
        schedule.forEach(input => {
            const epoch = parseInt(input.value);
            if (!isNaN(epoch) && epoch > totalEpochs) {
                hasInvalidEpochs = true;
                input.classList.add('is-invalid');
            } else {
                input.classList.remove('is-invalid');
            }
        });

        if (hasInvalidEpochs) {
            console.warn('Некоторые правила заданы для эпох, превышающих общее количество эпох обучения');
        }

        return !hasInvalidEpochs;
    }

    static showLoading(button, spinner, text) {
        if (button) {
            button.disabled = true;
            button.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> ${text}`;
        }
        if (spinner) {
            spinner.style.display = 'block';
        }
    }
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM загружен, инициализация приложения...');

    // Инициализация менеджера весов
    const weightsManager = new LossWeightsManager();

    // Загрузка существующей конфигурации если есть
    const existingConfig = document.getElementById('loss_weights_config')?.value;
    if (existingConfig) {
        console.log('Загружаем существующую конфигурацию:', existingConfig);
        weightsManager.loadConfig(existingConfig);
    }

    // Инициализация менеджера форм
    FormManager.init();

    // Сохраняем менеджер в глобальной области для отладки
    window.weightsManager = weightsManager;

    console.log('Приложение инициализировано');
});
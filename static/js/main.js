

class CarbonPlayground {
    constructor() {
        this.initializeElements();
        this.setupEventListeners();
        this.loadInitialData();
        this.currentPrediction = null;
        this.playgroundData = null;
        this.samplePrompts = [];
        
        // Animation and interaction state
        this.isAnimating = false;
        this.chartInstances = {};
        
        console.log('🌱 Carbon Prompt Playground initialized!');
    }

    /**
     * Initialize DOM elements
     */
    initializeElements() {
        // Form elements
        this.promptText = document.getElementById('prompt-text');
        this.promptType = document.getElementById('prompt-type');
        this.lengthType = document.getElementById('length-type');
        this.complexity = document.getElementById('complexity');
        this.tokenLength = document.getElementById('token-length');
        this.inferenceTime = document.getElementById('inference-time');
        this.energy = document.getElementById('energy');
        
        // Display elements
        this.tokenDisplay = document.getElementById('token-display');
        this.timeDisplay = document.getElementById('time-display');
        this.energyDisplay = document.getElementById('energy-display');
        
        // Metric cards
        this.carbonValue = document.getElementById('carbon-value');
        this.efficiencyValue = document.getElementById('efficiency-value');
        this.carbonPerToken = document.getElementById('carbon-per-token');
        
        // Buttons
        this.predictBtn = document.getElementById('predict-btn');
        this.randomBtn = document.getElementById('random-btn');
        this.optimizeBtn = document.getElementById('optimize-btn');
        this.resetBtn = document.getElementById('reset-btn');
        this.helpBtn = document.getElementById('help-btn');
        this.samplesBtn = document.getElementById('samples-btn');
        this.toggleInfoBtn = document.getElementById('toggle-info');
        this.toggleGuideBtn = document.getElementById('toggle-guide');
        
        // Panels and modals
        this.infoPanel = document.getElementById('info-panel');
        this.predictionDetails = document.getElementById('prediction-details');
        this.efficiencyTips = document.getElementById('efficiency-tips');
        this.environmentalImpact = document.getElementById('environmental-impact');
        this.helpModal = document.getElementById('help-modal');
        this.samplesModal = document.getElementById('samples-modal');
        this.errorModal = document.getElementById('error-modal');
        this.errorMessage = document.getElementById('error-message');
        this.loadingOverlay = document.getElementById('loading-overlay');
        
        // Chart containers
        this.carbonChart = document.getElementById('carbon-chart');
        this.energyCarbonChart = document.getElementById('energy-carbon-chart');
        this.featureImportanceChart = document.getElementById('feature-importance-chart');
        this.comparisonChart = document.getElementById('comparison-chart');
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Slider updates
        this.tokenLength.addEventListener('input', () => {
            this.tokenDisplay.textContent = this.tokenLength.value;
            this.autoPredict();
        });
        
        this.inferenceTime.addEventListener('input', () => {
            this.timeDisplay.textContent = this.inferenceTime.value;
            this.autoPredict();
        });
        
        this.energy.addEventListener('input', () => {
            this.energyDisplay.textContent = this.energy.value;
            this.autoPredict();
        });
        
        // Form changes
        [this.promptText, this.promptType, this.lengthType, this.complexity].forEach(element => {
            element.addEventListener('change', () => this.autoPredict());
        });
        
        this.promptText.addEventListener('input', this.debounce(() => this.autoPredict(), 500));
        
        // Buttons
        this.predictBtn.addEventListener('click', () => this.makePrediction());
        this.randomBtn.addEventListener('click', () => this.loadRandomExample());
        this.optimizeBtn.addEventListener('click', () => this.optimizeForLowCarbon());
        this.resetBtn.addEventListener('click', () => this.resetForm());
        
        // FAB buttons
        this.helpBtn.addEventListener('click', () => this.showModal('help'));
        this.samplesBtn.addEventListener('click', () => this.showModal('samples'));
        this.toggleInfoBtn.addEventListener('click', () => this.toggleInfoPanel());
        
        // Modal close buttons
        document.querySelectorAll('.close').forEach(closeBtn => {
            closeBtn.addEventListener('click', (e) => {
                const modal = e.target.closest('.modal');
                if (modal) this.hideModal(modal.id);
            });
        });
        
        // Close modals on outside click
        window.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                this.hideModal(e.target.id);
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                this.makePrediction();
            }
            if (e.key === 'Escape') {
                document.querySelectorAll('.modal').forEach(modal => {
                    if (modal.style.display === 'block') {
                        this.hideModal(modal.id);
                    }
                });
            }
        });
    }

    /**
     * Load initial data and setup
     */
    async loadInitialData() {
        try {
            // Load model info
            const modelInfo = await this.fetchAPI('/api/model_info');
            console.log('Model info loaded:', modelInfo);
            
            // Load playground data for visualizations
            this.playgroundData = await this.fetchAPI('/api/playground_data');
            console.log('Playground data loaded');
            
            // Load sample prompts
            const samplesData = await this.fetchAPI('/api/sample_prompts');
            this.samplePrompts = samplesData.samples || [];
            console.log('Sample prompts loaded');
            
            // Initialize charts
            this.initializeCharts();
            
            // Load a random example to start
            this.loadRandomExample();
            
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showError('Failed to load initial data. Please refresh the page.');
        }
    }

    /**
     * Initialize all charts
     */
    initializeCharts() {
        this.initializeCarbonChart();
        this.initializeEnergyCarbonChart();
        this.initializeFeatureImportanceChart();
        this.initializeComparisonChart();
    }

    /**
     * Initialize the main carbon emission chart
     */
    initializeCarbonChart() {
        const container = this.carbonChart;
        container.innerHTML = '';
        
        const margin = { top: 20, right: 30, bottom: 40, left: 60 };
        const width = container.clientWidth - margin.left - margin.right;
        const height = container.clientHeight - margin.top - margin.bottom;
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', container.clientWidth)
            .attr('height', container.clientHeight);
        
        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        // Add background gradient
        const defs = svg.append('defs');
        const gradient = defs.append('linearGradient')
            .attr('id', 'carbonGradient')
            .attr('gradientUnits', 'userSpaceOnUse')
            .attr('x1', 0).attr('y1', height)
            .attr('x2', 0).attr('y2', 0);
        
        gradient.append('stop')
            .attr('offset', '0%')
            .attr('stop-color', '#4CAF50')
            .attr('stop-opacity', 0.1);
        
        gradient.append('stop')
            .attr('offset', '100%')
            .attr('stop-color', '#4CAF50')
            .attr('stop-opacity', 0.5);
        
        // Initialize with placeholder
        g.append('text')
            .attr('x', width / 2)
            .attr('y', height / 2)
            .attr('text-anchor', 'middle')
            .attr('class', 'chart-placeholder')
            .style('font-size', '16px')
            .style('fill', '#757575')
            .text('Make a prediction to see real-time carbon emission data');
        
        this.chartInstances.carbonChart = { svg, g, width, height, margin };
    }

    /**
     * Initialize energy-carbon relationship chart
     */
    initializeEnergyCarbonChart() {
        const container = this.energyCarbonChart;
        container.innerHTML = '';
        
        if (!this.playgroundData || !this.playgroundData.energy_carbon_relationship) {
            container.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #757575;">Loading data...</div>';
            return;
        }
        
        const margin = { top: 20, right: 30, bottom: 40, left: 60 };
        const width = container.clientWidth - margin.left - margin.right;
        const height = container.clientHeight - margin.top - margin.bottom;
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', container.clientWidth)
            .attr('height', container.clientHeight);
        
        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        const data = this.playgroundData.energy_carbon_relationship;
        const points = data.energy.map((energy, i) => ({
            energy: energy,
            carbon: data.carbon[i]
        }));
        
        // Scales
        const xScale = d3.scaleLinear()
            .domain(d3.extent(points, d => d.energy))
            .range([0, width]);
        
        const yScale = d3.scaleLinear()
            .domain(d3.extent(points, d => d.carbon))
            .range([height, 0]);
        
        // Add axes
        g.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(xScale).tickFormat(d3.format('.3f')));
        
        g.append('g')
            .call(d3.axisLeft(yScale).tickFormat(d3.format('.6f')));
        
        // Add axis labels
        g.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', 0 - margin.left)
            .attr('x', 0 - (height / 2))
            .attr('dy', '1em')
            .style('text-anchor', 'middle')
            .style('font-size', '12px')
            .style('fill', '#757575')
            .text('Carbon Emission (g CO₂)');
        
        g.append('text')
            .attr('transform', `translate(${width / 2}, ${height + margin.bottom})`)
            .style('text-anchor', 'middle')
            .style('font-size', '12px')
            .style('fill', '#757575')
            .text('Energy Consumption (J)');
        
        // Add trend line (perfect linear relationship)
        const line = d3.line()
            .x(d => xScale(d.energy))
            .y(d => yScale(d.carbon));
        
        g.append('path')
            .datum(points.sort((a, b) => a.energy - b.energy))
            .attr('fill', 'none')
            .attr('stroke', '#2E7D32')
            .attr('stroke-width', 2)
            .attr('d', line);
        
        // Add data points
        g.selectAll('.data-point')
            .data(points)
            .enter().append('circle')
            .attr('class', 'data-point')
            .attr('cx', d => xScale(d.energy))
            .attr('cy', d => yScale(d.carbon))
            .attr('r', 3)
            .attr('fill', '#4CAF50')
            .attr('opacity', 0.7);
    }

    /**
     * Initialize feature importance chart
     */
    initializeFeatureImportanceChart() {
        const container = this.featureImportanceChart;
        container.innerHTML = '';
        
        const features = [
            { name: 'Energy', importance: 0.9611 },
            { name: 'Inference Time', importance: 0.0286 },
            { name: 'Energy/Token', importance: 0.0023 },
            { name: 'Word Count', importance: 0.0012 },
            { name: 'Token Length', importance: 0.0005 },
            { name: 'Other Features', importance: 0.0063 }
        ];
        
        const margin = { top: 20, right: 30, bottom: 40, left: 100 };
        const width = container.clientWidth - margin.left - margin.right;
        const height = container.clientHeight - margin.top - margin.bottom;
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', container.clientWidth)
            .attr('height', container.clientHeight);
        
        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        // Scales
        const xScale = d3.scaleLinear()
            .domain([0, d3.max(features, d => d.importance)])
            .range([0, width]);
        
        const yScale = d3.scaleBand()
            .domain(features.map(d => d.name))
            .range([0, height])
            .padding(0.1);
        
        // Add bars
        g.selectAll('.bar')
            .data(features)
            .enter().append('rect')
            .attr('class', 'bar')
            .attr('x', 0)
            .attr('y', d => yScale(d.name))
            .attr('width', d => xScale(d.importance))
            .attr('height', yScale.bandwidth())
            .attr('fill', (d, i) => d3.interpolateGreens(0.3 + (i * 0.1)))
            .attr('rx', 3);
        
        // Add labels
        g.selectAll('.bar-label')
            .data(features)
            .enter().append('text')
            .attr('class', 'bar-label')
            .attr('x', d => xScale(d.importance) + 5)
            .attr('y', d => yScale(d.name) + yScale.bandwidth() / 2)
            .attr('dy', '0.35em')
            .style('font-size', '11px')
            .style('fill', '#333')
            .text(d => `${(d.importance * 100).toFixed(1)}%`);
        
        // Add y-axis
        g.append('g')
            .call(d3.axisLeft(yScale))
            .style('font-size', '11px');
    }

    /**
     * Initialize comparison chart
     */
    initializeComparisonChart() {
        const container = this.comparisonChart;
        container.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #757575;">Make a prediction to see comparison data</div>';
    }

    /**
     * Make a carbon emission prediction
     */
    async makePrediction() {
        if (this.isAnimating) return;
        
        const data = this.gatherFormData();
        
        if (!this.validateFormData(data)) {
            return;
        }
        
        this.showLoading();
        this.isAnimating = true;
        
        try {
            console.log('Making prediction with data:', data);
            
            const result = await this.fetchAPI('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            
            console.log('Prediction result:', result);
            
            this.currentPrediction = result;
            this.updateUI(result);
            
            // Try to update charts, but don't fail if charts aren't ready
            try {
                this.updateCharts(result);
                console.log('Charts updated successfully');
            } catch (chartError) {
                console.warn('Chart update failed (non-critical):', chartError.message);
                // Don't throw the error - charts are secondary to the main prediction
            }
            
            // Show success animation
            this.showSuccessAnimation();
            
            console.log('Prediction completed successfully');
            
        } catch (error) {
            console.error('Prediction error:', error);
            console.error('Error stack:', error.stack);
            this.showError(`Failed to make prediction: ${error.message}`);
        } finally {
            this.hideLoading();
            this.isAnimating = false;
        }
    }

    /**
     * Auto-predict with debouncing for real-time updates
     */
    autoPredict() {
        if (this.promptText.value.trim().length > 10) {
            clearTimeout(this.autoPredictTimeout);
            this.autoPredictTimeout = setTimeout(() => {
                this.makePrediction();
            }, 1000);
        }
    }

    /**
     * Gather form data
     */
    gatherFormData() {
        return {
            prompt_text: this.promptText.value.trim(),
            prompt_type: this.promptType.value,
            length_type: this.lengthType.value,
            prompt_complexity: this.complexity.value,
            token_length: parseInt(this.tokenLength.value),
            inference_time: parseFloat(this.inferenceTime.value),
            energy: parseFloat(this.energy.value)
        };
    }

    /**
     * Validate form data
     */
    validateFormData(data) {
        if (!data.prompt_text) {
            this.showError('Please enter a prompt text');
            return false;
        }
        
        if (data.prompt_text.length < 5) {
            this.showError('Prompt text must be at least 5 characters long');
            return false;
        }
        
        if (data.token_length < 1 || data.token_length > 200) {
            this.showError('Token length must be between 1 and 200');
            return false;
        }
        
        if (data.inference_time < 0.1 || data.inference_time > 100) {
            this.showError('Inference time must be between 0.1 and 100 seconds');
            return false;
        }
        
        if (data.energy < 0.001 || data.energy > 1.0) {
            this.showError('Energy must be between 0.001 and 1.0 joules');
            return false;
        }
        
        return true;
    }

    /**
     * Update UI with prediction results
     */
    updateUI(result) {
        try {
            console.log('🎨 Updating UI with result:', result);
            
            // Update metric cards with animation
            this.animateValue(this.carbonValue, parseFloat(result.predicted_carbon_emission).toFixed(8));
            this.animateValue(this.efficiencyValue, result.efficiency_rating);
            this.animateValue(this.carbonPerToken, parseFloat(result.carbon_per_token).toFixed(8));
            
            // Update analysis panel
            this.updateAnalysisPanel(result);
            
            console.log('UI updated successfully');
        } catch (error) {
            console.error('updateUI Error:', error);
            throw error;
        }
    }

    updateAnalysisPanel(result) {
        try {
            // Update carbon emission
            const carbonElement = document.getElementById('analysis-carbon');
            if (carbonElement) {
                carbonElement.textContent = `${parseFloat(result.predicted_carbon_emission).toFixed(8)} g CO₂`;
            }
            
            // Update energy consumption
            const energyElement = document.getElementById('analysis-energy');
            if (energyElement) {
                energyElement.textContent = `${result.energy_consumption.toFixed(6)} J`;
            }
            
            // Update efficiency score
            const efficiencyElement = document.getElementById('analysis-efficiency');
            if (efficiencyElement) {
                efficiencyElement.textContent = result.efficiency_score.toFixed(2);
            }
            
            // Update model info
            const modelElement = document.getElementById('analysis-model');
            if (modelElement) {
                modelElement.textContent = result.model_used;
            }
            
            // Update confidence
            const confidenceElement = document.getElementById('analysis-confidence');
            if (confidenceElement) {
                confidenceElement.textContent = result.confidence;
            }
            
            console.log('Analysis panel updated');
        } catch (error) {
            console.error('updateAnalysisPanel Error:', error);
            throw error;
        }
    }

    /**
     * Update prediction details
     */
    updatePredictionDetails(result) {
        const html = `
            <div class="prediction-summary">
                <div class="summary-item">
                    <strong>Carbon Emission:</strong> ${parseFloat(result.predicted_carbon_emission).toFixed(8)} g CO₂
                </div>
                <div class="summary-item">
                    <strong>Energy Consumption:</strong> ${result.energy_consumption.toFixed(6)} J
                </div>
                <div class="summary-item">
                    <strong>Efficiency Score:</strong> ${result.efficiency_score.toFixed(2)}
                </div>
                <div class="summary-item">
                    <strong>Confidence:</strong> ${result.confidence}
                </div>
                <div class="summary-item">
                    <strong>Model:</strong> ${result.model_used}
                </div>
            </div>
            
            <div class="environmental-metrics">
                <h4>Environmental Metrics</h4>
                <div class="metric-row">
                    <span>Impact Level:</span>
                    <span class="impact-badge" style="background-color: ${result.environmental_impact.color}">
                        ${result.environmental_impact.level}
                    </span>
                </div>
                <div class="metric-row">
                    <span>Tree Equivalent:</span>
                    <span>${result.environmental_impact.tree_equivalent.toFixed(6)} trees/day</span>
                </div>
            </div>
        `;
        
        this.predictionDetails.innerHTML = html;
        this.predictionDetails.classList.add('fade-in');
    }

    /**
     * Update efficiency tips
     */
    updateEfficiencyTips(result) {
        const tips = result.efficiency_tips || [];
        
        if (tips.length === 0) {
            this.efficiencyTips.innerHTML = '<p>Your prompt is already quite efficient!</p>';
            return;
        }
        
        const html = '<ul>' + tips.map(tip => `<li>${tip}</li>`).join('') + '</ul>';
        this.efficiencyTips.innerHTML = html;
        this.efficiencyTips.classList.add('fade-in');
    }

    /**
     * Update environmental impact
     */
    updateEnvironmentalImpact(result) {
        const impact = result.environmental_impact;
        
        const html = `
            <div class="impact-summary">
                <div class="impact-level" style="border-left-color: ${impact.color}">
                    <strong>Impact Level:</strong> ${impact.level}
                </div>
                <div class="impact-comparison">
                    <p>${impact.comparison}</p>
                </div>
                <div class="impact-metrics">
                    <div>CO₂ Equivalent: ${(impact.kg_co2 * 1000).toFixed(3)} mg</div>
                    <div>Environmental Cost: ${impact.level} impact</div>
                </div>
            </div>
        `;
        
        this.environmentalImpact.innerHTML = html;
        this.environmentalImpact.classList.add('fade-in');
    }

    /**
     * Update charts with new prediction
     */
    updateCharts(result) {
        try {
            console.log('Starting chart updates...');
            
            // Update each chart with error handling
            try {
                this.updateCarbonChart(result);
                console.log('Carbon chart updated');
            } catch (error) {
                console.warn('Carbon chart update failed:', error.message);
            }
            
            try {
                this.updateComparisonChart(result);
                console.log('Comparison chart updated');
            } catch (error) {
                console.warn('Comparison chart update failed:', error.message);
            }
            
            console.log('All charts update process completed');
        } catch (error) {
            console.error('Chart update process failed:', error);
            throw error;
        }
    }

    /**
     * Update main carbon chart
     */
    updateCarbonChart(result) {
        const chartData = this.chartInstances.carbonChart;
        if (!chartData) return;
        
        const { g, width, height } = chartData;
        
        // Clear previous content
        g.selectAll('*').remove();
        
        // Create data points for visualization
        const currentTime = Date.now();
        const timeRange = 60000; // 1 minute
        const dataPoints = [];
        
        // Add some historical context (simulated)
        for (let i = 0; i < 20; i++) {
            dataPoints.push({
                time: currentTime - timeRange + (i * timeRange / 20),
                carbon: (Math.random() * 0.0002) + 0.00005, // Random historical data
                isPrediction: false
            });
        }
        
        // Add current prediction
        dataPoints.push({
            time: currentTime,
            carbon: result.predicted_carbon_emission,
            isPrediction: true
        });
        
        // Scales
        const xScale = d3.scaleTime()
            .domain(d3.extent(dataPoints, d => d.time))
            .range([0, width]);
        
        const yScale = d3.scaleLinear()
            .domain([0, d3.max(dataPoints, d => d.carbon) * 1.2])
            .range([height, 0]);
        
        // Add axes
        g.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(xScale).tickFormat(d3.timeFormat('%H:%M:%S')));
        
        g.append('g')
            .call(d3.axisLeft(yScale).tickFormat(d3.format('.6f')));
        
        // Add area chart
        const area = d3.area()
            .x(d => xScale(d.time))
            .y0(height)
            .y1(d => yScale(d.carbon))
            .curve(d3.curveMonotoneX);
        
        g.append('path')
            .datum(dataPoints.filter(d => !d.isPrediction))
            .attr('fill', 'url(#carbonGradient)')
            .attr('d', area);
        
        // Add line
        const line = d3.line()
            .x(d => xScale(d.time))
            .y(d => yScale(d.carbon))
            .curve(d3.curveMonotoneX);
        
        g.append('path')
            .datum(dataPoints.filter(d => !d.isPrediction))
            .attr('fill', 'none')
            .attr('stroke', '#2E7D32')
            .attr('stroke-width', 2)
            .attr('d', line);
        
        // Highlight current prediction
        const predictionPoint = dataPoints.find(d => d.isPrediction);
        g.append('circle')
            .attr('cx', xScale(predictionPoint.time))
            .attr('cy', yScale(predictionPoint.carbon))
            .attr('r', 8)
            .attr('fill', '#1976D2')
            .attr('stroke', '#fff')
            .attr('stroke-width', 3)
            .style('animation', 'pulse 2s infinite');
        
        // Add labels
        g.append('text')
            .attr('x', width / 2)
            .attr('y', -5)
            .attr('text-anchor', 'middle')
            .style('font-size', '14px')
            .style('font-weight', 'bold')
            .style('fill', '#333')
            .text('Real-time Carbon Emission Prediction');
    }

    /**
     * Update comparison chart
     */
    updateComparisonChart(result) {
        const container = this.comparisonChart;
        container.innerHTML = '';
        
        const comparisonData = result.comparison_data;
        if (!comparisonData) return;
        
        const data = [
            { label: 'Your Prompt', carbon: result.predicted_carbon_emission, color: '#1976D2' },
            { label: 'Dataset Average', carbon: 0.0001117, color: '#757575' },
            { label: 'Low Impact', carbon: 0.00006, color: '#4CAF50' },
            { label: 'High Impact', carbon: 0.00025, color: '#F44336' }
        ];
        
        const margin = { top: 20, right: 30, bottom: 40, left: 80 };
        const width = container.clientWidth - margin.left - margin.right;
        const height = container.clientHeight - margin.top - margin.bottom;
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', container.clientWidth)
            .attr('height', container.clientHeight);
        
        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        // Scales
        const xScale = d3.scaleLinear()
            .domain([0, d3.max(data, d => d.carbon)])
            .range([0, width]);
        
        const yScale = d3.scaleBand()
            .domain(data.map(d => d.label))
            .range([0, height])
            .padding(0.2);
        
        // Add bars
        g.selectAll('.comparison-bar')
            .data(data)
            .enter().append('rect')
            .attr('class', 'comparison-bar')
            .attr('x', 0)
            .attr('y', d => yScale(d.label))
            .attr('width', d => xScale(d.carbon))
            .attr('height', yScale.bandwidth())
            .attr('fill', d => d.color)
            .attr('rx', 3)
            .style('opacity', 0.8);
        
        // Add value labels
        g.selectAll('.comparison-label')
            .data(data)
            .enter().append('text')
            .attr('class', 'comparison-label')
            .attr('x', d => xScale(d.carbon) + 5)
            .attr('y', d => yScale(d.label) + yScale.bandwidth() / 2)
            .attr('dy', '0.35em')
            .style('font-size', '11px')
            .style('fill', '#333')
            .text(d => d.carbon.toFixed(6));
        
        // Add y-axis
        g.append('g')
            .call(d3.axisLeft(yScale))
            .style('font-size', '11px');
    }

    /**
     * Load a random example
     */
    loadRandomExample() {
        if (this.samplePrompts.length === 0) {
            // Use a default example
            this.loadDefaultExample();
            return;
        }
        
        const randomSample = this.samplePrompts[Math.floor(Math.random() * this.samplePrompts.length)];
        
        this.promptText.value = randomSample.prompt_text;
        this.promptType.value = randomSample.prompt_type;
        this.lengthType.value = randomSample.length_type;
        this.complexity.value = randomSample.prompt_complexity;
        this.tokenLength.value = randomSample.token_length;
        this.inferenceTime.value = randomSample.inference_time.toFixed(1);
        this.energy.value = randomSample.energy.toFixed(3);
        
        // Update displays
        this.tokenDisplay.textContent = randomSample.token_length;
        this.timeDisplay.textContent = randomSample.inference_time.toFixed(1);
        this.energyDisplay.textContent = randomSample.energy.toFixed(3);
        
        // Make prediction
        setTimeout(() => this.makePrediction(), 500);
    }

    /**
     * Load default example
     */
    loadDefaultExample() {
        this.promptText.value = 'Classify this requirement as functional or non-functional: "The system shall respond within 2 seconds"';
        this.promptType.value = 'zero_shot';
        this.lengthType.value = 'short';
        this.complexity.value = 'low';
        this.tokenLength.value = 20;
        this.inferenceTime.value = 15.0;
        this.energy.value = 0.095;
        
        this.tokenDisplay.textContent = '20';
        this.timeDisplay.textContent = '15.0';
        this.energyDisplay.textContent = '0.095';
    }

    /**
     * Optimize for low carbon emission
     */
    optimizeForLowCarbon() {
        // Suggest optimizations
        const currentEnergy = parseFloat(this.energy.value);
        const currentTime = parseFloat(this.inferenceTime.value);
        const currentTokens = parseInt(this.tokenLength.value);
        
        // Reduce energy and time by 20-30%
        const optimizedEnergy = Math.max(0.01, currentEnergy * 0.7);
        const optimizedTime = Math.max(1, currentTime * 0.8);
        const optimizedTokens = Math.max(5, Math.floor(currentTokens * 0.85));
        
        // Update form
        this.energy.value = optimizedEnergy.toFixed(3);
        this.inferenceTime.value = optimizedTime.toFixed(1);
        this.tokenLength.value = optimizedTokens;
        
        // Update displays
        this.energyDisplay.textContent = optimizedEnergy.toFixed(3);
        this.timeDisplay.textContent = optimizedTime.toFixed(1);
        this.tokenDisplay.textContent = optimizedTokens;
        
        // Switch to lower complexity if possible
        if (this.complexity.value === 'high') {
            this.complexity.value = 'medium';
        } else if (this.complexity.value === 'medium') {
            this.complexity.value = 'low';
        }
        
        // Suggest shorter prompt
        if (this.promptText.value.length > 100) {
            this.showInfo('Consider shortening your prompt for better efficiency!');
        }
        
        // Make prediction with optimized values
        setTimeout(() => this.makePrediction(), 500);
    }

    /**
     * Reset form to defaults
     */
    resetForm() {
        this.promptText.value = '';
        this.promptType.value = 'zero_shot';
        this.lengthType.value = 'short';
        this.complexity.value = 'low';
        this.tokenLength.value = 25;
        this.inferenceTime.value = 15.5;
        this.energy.value = 0.095;
        
        this.tokenDisplay.textContent = '25';
        this.timeDisplay.textContent = '15.5';
        this.energyDisplay.textContent = '0.095';
        
        // Reset UI
        this.carbonValue.textContent = '0.000000';
        this.efficiencyValue.textContent = '-';
        this.carbonPerToken.textContent = '0.000000';
        
        this.predictionDetails.innerHTML = '<p>Enter a prompt and adjust parameters to see predictions...</p>';
        this.efficiencyTips.innerHTML = '<p>Tips will appear here after making a prediction...</p>';
        this.environmentalImpact.innerHTML = '<p>Environmental impact analysis will be shown here...</p>';
        
        // Clear charts
        this.initializeCharts();
    }

    /**
     * Show modal
     */
    showModal(modalType) {
        if (modalType === 'help') {
            this.helpModal.style.display = 'block';
        } else if (modalType === 'samples') {
            this.loadSamplePrompts();
            this.samplesModal.style.display = 'block';
        }
    }

    /**
     * Hide modal
     */
    hideModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.style.display = 'none';
        }
    }

    /**
     * Toggle info panel
     */
    toggleInfoPanel() {
        this.infoPanel.classList.toggle('open');
    }

    /**
     * Load sample prompts into modal
     */
    loadSamplePrompts() {
        const container = document.getElementById('sample-prompts-list');
        
        if (this.samplePrompts.length === 0) {
            container.innerHTML = '<p>No sample prompts available. Please try refreshing the page.</p>';
            return;
        }
        
        const html = this.samplePrompts.map((sample, index) => `
            <div class="sample-prompt-item" data-index="${index}">
                <div class="sample-header">
                    <strong>${sample.prompt_type}</strong> - ${sample.prompt_complexity} complexity
                </div>
                <div class="sample-text">${sample.prompt_text}</div>
                <div class="sample-metrics">
                    <span>Tokens: ${sample.token_length}</span> |
                    <span>Time: ${sample.inference_time.toFixed(1)}s</span> |
                    <span>Energy: ${sample.energy.toFixed(3)}J</span> |
                    <span>Actual CO₂: ${sample.actual_carbon.toFixed(6)}g</span>
                </div>
                <button class="btn btn-primary sample-load-btn" onclick="playground.loadSample(${index})">
                    Load This Example
                </button>
            </div>
        `).join('');
        
        container.innerHTML = html;
    }

    /**
     * Load specific sample
     */
    loadSample(index) {
        const sample = this.samplePrompts[index];
        if (!sample) return;
        
        this.promptText.value = sample.prompt_text;
        this.promptType.value = sample.prompt_type;
        this.lengthType.value = sample.length_type;
        this.complexity.value = sample.prompt_complexity;
        this.tokenLength.value = sample.token_length;
        this.inferenceTime.value = sample.inference_time.toFixed(1);
        this.energy.value = sample.energy.toFixed(3);
        
        this.tokenDisplay.textContent = sample.token_length;
        this.timeDisplay.textContent = sample.inference_time.toFixed(1);
        this.energyDisplay.textContent = sample.energy.toFixed(3);
        
        this.hideModal('samples-modal');
        
        setTimeout(() => this.makePrediction(), 500);
    }

    /**
     * Utility functions
     */
    
    async fetchAPI(endpoint, options = {}) {
        try {
            console.log('Making API request to:', endpoint);
            const response = await fetch(endpoint, options);
            console.log('Response status:', response.status, response.statusText);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                console.error('API Error Response:', errorData);
                throw new Error(errorData.error || `API request failed: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('API Response data:', data);
            return data;
        } catch (error) {
            console.error('fetchAPI Error:', error);
            throw error;
        }
    }

    showLoading() {
        this.loadingOverlay.style.display = 'flex';
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }

    showError(message) {
        // Show styled error modal instead of alert
        this.errorMessage.textContent = message;
        this.errorModal.style.display = 'block';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            this.closeErrorModal();
        }, 5000);
    }

    closeErrorModal() {
        if (this.errorModal) {
            this.errorModal.style.display = 'none';
        }
    }

    showInfo(message) {
        // Simple info display - could be enhanced with a toast system
        alert('ℹ️ ' + message);
    }

    showSuccessAnimation() {
        // Add success animation to predict button
        this.predictBtn.style.background = '#4CAF50';
        setTimeout(() => {
            this.predictBtn.style.background = '';
        }, 1000);
    }

    animateValue(element, newValue) {
        element.style.transform = 'scale(1.1)';
        element.style.color = '#1976D2';
        setTimeout(() => {
            element.textContent = newValue;
            element.style.transform = 'scale(1)';
            element.style.color = '';
        }, 150);
    }

    toggleUsageGuide() {
        const content = document.getElementById('usage-content');
        const btn = this.toggleGuideBtn;
        
        if (content.classList.contains('collapsed')) {
            content.classList.remove('collapsed');
            btn.innerHTML = '<i class="fas fa-eye-slash"></i> Hide Guide';
        } else {
            content.classList.add('collapsed');
            btn.innerHTML = '<i class="fas fa-eye"></i> Show Guide';
        }
    }

    focusPromptInput() {
        if (this.promptText) {
            this.promptText.focus();
            this.promptText.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }

    setupUsageGuideButtons() {
        // Make sure all usage guide buttons are properly connected
        console.log('Setting up usage guide buttons...');
        
        // Try Example button
        const tryExampleBtn = document.querySelector('button[onclick="playground.loadRandomExample()"]');
        if (tryExampleBtn) {
            tryExampleBtn.onclick = () => {
                console.log('Try Example button clicked');
                this.loadRandomExample();
            };
        }
        
        // Start Here button  
        const startHereBtn = document.querySelector('button[onclick="playground.focusPromptInput()"]');
        if (startHereBtn) {
            startHereBtn.onclick = () => {
                console.log('Start Here button clicked');
                this.focusPromptInput();
            };
        }
        
        // Hide/Show Guide button
        const toggleBtn = document.getElementById('toggle-guide');
        if (toggleBtn) {
            toggleBtn.onclick = () => {
                console.log('Toggle Guide button clicked');
                this.toggleUsageGuide();
            };
        }
        
        console.log('Usage guide buttons setup completed');
    }

    debounce(func, delay) {
        let timeout;
        return function(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), delay);
        };
    }
}

// Initialize the playground when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.playground = new CarbonPlayground();
});

// Add some CSS for sample prompts
const sampleStyles = `
    .sample-prompt-item {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s;
    }
    
    .sample-prompt-item:hover {
        border-color: #2E7D32;
    }
    
    .sample-header {
        color: #2E7D32;
        font-size: 14px;
        margin-bottom: 0.5rem;
    }
    
    .sample-text {
        background: #f5f5f5;
        padding: 0.5rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        font-family: monospace;
        font-size: 13px;
    }
    
    .sample-metrics {
        font-size: 12px;
        color: #757575;
        margin-bottom: 1rem;
    }
    
    .sample-load-btn {
        font-size: 12px;
        padding: 0.5rem 1rem;
    }
    
    .prediction-summary, .environmental-metrics {
        margin-bottom: 1rem;
    }
    
    .summary-item, .metric-row {
        padding: 0.5rem 0;
        border-bottom: 1px solid #f0f0f0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .impact-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        color: white;
        font-size: 12px;
        font-weight: bold;
    }
    
    .impact-summary {
        border-left: 4px solid #4CAF50;
        padding-left: 1rem;
    }
    
    .impact-level {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
`;

// Inject styles
const styleSheet = document.createElement('style');
styleSheet.textContent = sampleStyles;
document.head.appendChild(styleSheet);
// Cricket Analysis Application JavaScript

class CricketAnalysisApp {
    constructor() {
        this.currentFileId = null;
        this.analysisTypes = ['bowling', 'batting'];
        this.uploadedVideos = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadUploadedVideos();
        this.checkHealth();
    }

    setupEventListeners() {
        // File upload
        const fileInput = document.getElementById('videoFile');
        const uploadArea = document.getElementById('uploadArea');
        const uploadBtn = document.getElementById('uploadBtn');

        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }

        if (uploadArea) {
            uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
            uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
            uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
            uploadArea.addEventListener('click', () => fileInput?.click());
        }

        if (uploadBtn) {
            uploadBtn.addEventListener('click', () => this.uploadVideo());
        }

        // Analysis buttons
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('analyze-btn')) {
                const fileId = e.target.dataset.fileId;
                const analysisType = e.target.dataset.analysisType;
                this.performAnalysis(fileId, analysisType);
            }
        });

        // Refresh videos button
        const refreshBtn = document.getElementById('refreshVideos');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadUploadedVideos());
        }
    }

    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            
            const healthStatus = document.getElementById('healthStatus');
            if (healthStatus) {
                if (data.status === 'healthy') {
                    healthStatus.innerHTML = '<span class="status-indicator status-success"></span>API Services Online';
                    healthStatus.className = 'text-success';
                } else {
                    healthStatus.innerHTML = '<span class="status-indicator status-danger"></span>API Services Offline';
                    healthStatus.className = 'text-danger';
                }
            }
        } catch (error) {
            console.error('Health check failed:', error);
            const healthStatus = document.getElementById('healthStatus');
            if (healthStatus) {
                healthStatus.innerHTML = '<span class="status-indicator status-danger"></span>API Services Offline';
                healthStatus.className = 'text-danger';
            }
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            document.getElementById('videoFile').files = files;
            this.handleFileSelect({ target: { files } });
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.displayFileInfo(file);
            document.getElementById('uploadBtn').disabled = false;
        }
    }

    displayFileInfo(file) {
        const fileInfo = document.getElementById('fileInfo');
        if (fileInfo) {
            const sizeInMB = (file.size / (1024 * 1024)).toFixed(2);
            fileInfo.innerHTML = `
                <div class="alert alert-info">
                    <strong>Selected File:</strong> ${file.name}<br>
                    <strong>Size:</strong> ${sizeInMB} MB<br>
                    <strong>Type:</strong> ${file.type}
                </div>
            `;
        }
    }

    async uploadVideo() {
        const fileInput = document.getElementById('videoFile');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showAlert('Please select a video file first.', 'warning');
            return;
        }

        // Validate file type
        const allowedTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/quicktime'];
        if (!allowedTypes.includes(file.type)) {
            this.showAlert('Please select a valid video file (MP4, AVI, MOV).', 'danger');
            return;
        }

        // Validate file size (500MB limit)
        if (file.size > 500 * 1024 * 1024) {
            this.showAlert('File size exceeds 500MB limit.', 'danger');
            return;
        }

        this.showProgress('Uploading video...');

        try {
            const formData = new FormData();
            formData.append('video', file);

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                this.currentFileId = data.file_id;
                this.showAlert(`Video uploaded successfully! File ID: ${data.file_id}`, 'success');
                this.showAnalysisOptions(data.file_id);
                this.loadUploadedVideos(); // Refresh the video list
            } else {
                this.showAlert(`Upload failed: ${data.error}`, 'danger');
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showAlert('Upload failed due to network error.', 'danger');
        } finally {
            this.hideProgress();
        }
    }

    showAnalysisOptions(fileId) {
        const analysisOptions = document.getElementById('analysisOptions');
        if (analysisOptions) {
            analysisOptions.innerHTML = `
                <div class="analysis-section">
                    <h5>Analysis Options</h5>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="analysis-card p-3">
                                <h6>🏏 Bowling Analysis</h6>
                                <p>Track ball trajectory, speed, and delivery accuracy</p>
                                <button class="btn btn-primary analyze-btn" data-file-id="${fileId}" data-analysis-type="bowling">
                                    Analyze Bowling
                                </button>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="analysis-card p-3">
                                <h6>🏃 Batting Analysis</h6>
                                <p>Analyze batting technique and body mechanics</p>
                                <button class="btn btn-success analyze-btn" data-file-id="${fileId}" data-analysis-type="batting">
                                    Analyze Batting
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            analysisOptions.style.display = 'block';
        }
    }

    async performAnalysis(fileId, analysisType) {
        const button = document.querySelector(`[data-file-id="${fileId}"][data-analysis-type="${analysisType}"]`);
        const originalText = button.textContent;
        
        button.disabled = true;
        button.innerHTML = '<span class="loading-spinner"></span> Analyzing...';

        this.showProgress(`Performing ${analysisType} analysis...`);

        try {
            const response = await fetch(`/api/analyze/${analysisType}/${fileId}`, {
                method: 'POST'
            });

            const data = await response.json();

            if (response.ok) {
                this.displayAnalysisResults(data, analysisType);
                this.showAlert(`${analysisType.charAt(0).toUpperCase() + analysisType.slice(1)} analysis completed!`, 'success');
            } else {
                this.showAlert(`Analysis failed: ${data.error}`, 'danger');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            this.showAlert('Analysis failed due to network error.', 'danger');
        } finally {
            button.disabled = false;
            button.textContent = originalText;
            this.hideProgress();
        }
    }

    displayAnalysisResults(data, analysisType) {
        const resultsContainer = document.getElementById('resultsContainer');
        if (!resultsContainer) return;

        let resultsHTML = '';

        if (analysisType === 'bowling') {
            resultsHTML = this.generateBowlingResultsHTML(data);
        } else if (analysisType === 'batting') {
            resultsHTML = this.generateBattingResultsHTML(data);
        }

        resultsContainer.innerHTML = resultsHTML;
        resultsContainer.style.display = 'block';

        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }

    generateBowlingResultsHTML(data) {
        const metrics = data.performance_metrics || {};
        const speedAnalysis = metrics.speed_analysis || {};
        const accuracyAnalysis = metrics.accuracy_analysis || {};
        const trajectoryAnalysis = metrics.trajectory_analysis || {};
        const overallScores = data.overall_scores || {};
        const visualizations = data.visualizations || {};

        return `
            <div class="analysis-section">
                <h4>🏏 Bowling Analysis Results</h4>
                
                ${visualizations.pitch_plot ? `
                <div class="row mb-4">
                    <div class="col-12">
                        <div class="visualization-section">
                            <h5>Hawk-Eye Ball Trajectory Analysis</h5>
                            <div class="text-center">
                                <img src="data:image/png;base64,${visualizations.pitch_plot}" 
                                     class="img-fluid analysis-chart" 
                                     alt="Hawk-Eye Pitch Plot"
                                     style="max-width: 100%; border: 1px solid #ddd; border-radius: 8px;">
                            </div>
                            <p class="text-muted text-center mt-2">${visualizations.pitch_plot_description}</p>
                        </div>
                    </div>
                </div>
                ` : ''}
                
                <div class="row mb-4">
                    <div class="col-md-4">
                        <div class="metric-card text-center">
                            <h5>Average Speed</h5>
                            <div class="display-6 text-primary">${speedAnalysis.average_speed_kmh || 0} km/h</div>
                            <small class="text-muted">Category: ${speedAnalysis.speed_category || 'Unknown'}</small>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="metric-card text-center">
                            <h5>Accuracy Score</h5>
                            <div class="display-6 text-success">${Math.round(accuracyAnalysis.accuracy_score || 0)}/100</div>
                            <small class="text-muted">Line: ${accuracyAnalysis.line_consistency || 'Unknown'}</small>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="metric-card text-center">
                            <h5>Overall Score</h5>
                            <div class="display-6 text-info">${Math.round(overallScores.overall || 0)}/100</div>
                            <small class="text-muted">Performance Rating</small>
                        </div>
                    </div>
                </div>

                <div class="row">
                    <div class="col-md-6">
                        <div class="analysis-section">
                            <h6>Speed Analysis</h6>
                            <ul class="list-unstyled">
                                <li><strong>Peak Speed:</strong> ${speedAnalysis.max_speed_kmh || 0} km/h</li>
                                <li><strong>Consistency:</strong> ${speedAnalysis.speed_consistency || 'Unknown'}</li>
                                <li><strong>Category:</strong> ${speedAnalysis.speed_category || 'Unknown'}</li>
                            </ul>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="analysis-section">
                            <h6>Accuracy Assessment</h6>
                            <ul class="list-unstyled">
                                <li><strong>Line:</strong> ${accuracyAnalysis.line_consistency || 'Unknown'}</li>
                                <li><strong>Length:</strong> ${accuracyAnalysis.length_assessment || 'Unknown'}</li>
                                <li><strong>Accuracy Score:</strong> ${Math.round(accuracyAnalysis.accuracy_score || 0)}/100</li>
                            </ul>
                        </div>
                    </div>
                </div>

                ${data.recommendations ? `
                <div class="analysis-section">
                    <h6>Recommendations</h6>
                    <ul class="recommendation-list">
                        ${data.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                    </ul>
                </div>
                ` : ''}
            </div>
        `;
    }

    generateBattingResultsHTML(data) {
        const metrics = data.performance_metrics || {};
        const techniqueAnalysis = data.technique_analysis || {};
        const improvementPlan = data.improvement_plan || {};
        const aspectScores = metrics.aspect_scores || {};
        const visualizations = data.visualizations || {};

        return `
            <div class="analysis-section">
                <h4>🏃 Batting Analysis Results</h4>
                
                ${visualizations.labeled_frames ? `
                <div class="row mb-4">
                    <div class="col-12">
                        <div class="visualization-section">
                            <h5>Skeleton Tracking Analysis</h5>
                            <div class="text-center">
                                <img src="data:image/jpeg;base64,${visualizations.labeled_frames}" 
                                     class="img-fluid analysis-chart" 
                                     alt="Skeleton Tracking"
                                     style="max-width: 100%; border: 1px solid #ddd; border-radius: 8px;">
                            </div>
                            <p class="text-muted text-center mt-2">${visualizations.labeled_frames_description}</p>
                        </div>
                    </div>
                </div>
                ` : ''}

                ${visualizations.summary_plot ? `
                <div class="row mb-4">
                    <div class="col-12">
                        <div class="visualization-section">
                            <h5>Technique Analysis Charts</h5>
                            <div class="text-center">
                                <img src="data:image/png;base64,${visualizations.summary_plot}" 
                                     class="img-fluid analysis-chart" 
                                     alt="Batting Technique Charts"
                                     style="max-width: 100%; border: 1px solid #ddd; border-radius: 8px;">
                            </div>
                            <p class="text-muted text-center mt-2">${visualizations.summary_plot_description}</p>
                        </div>
                    </div>
                </div>
                ` : ''}
                
                <div class="row mb-4">
                    <div class="col-md-4">
                        <div class="metric-card text-center">
                            <h5>Technique Score</h5>
                            <div class="display-6 text-primary">${Math.round(metrics.overall_technique_score || 0)}/100</div>
                            <small class="text-muted">Grade: ${metrics.technique_grade || 'N/A'}</small>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="metric-card text-center">
                            <h5>Consistency</h5>
                            <div class="display-6 text-success">${Math.round(metrics.consistency_score || 0)}/100</div>
                            <small class="text-muted">Performance Level: ${metrics.performance_level || 'Unknown'}</small>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="metric-card text-center">
                            <h5>Detection Rate</h5>
                            <div class="display-6 text-info">${data.analysis_summary?.detection_rate || '0%'}</div>
                            <small class="text-muted">Analysis Quality: ${data.analysis_summary?.analysis_quality || 'Unknown'}</small>
                        </div>
                    </div>
                </div>

                ${Object.keys(aspectScores).length > 0 ? `
                <div class="analysis-section">
                    <h6>Technique Breakdown</h6>
                    <div class="technique-breakdown">
                        ${Object.entries(aspectScores).map(([aspect, score]) => `
                            <div class="metric-card text-center">
                                <h6>${aspect}</h6>
                                <div class="h4 ${this.getScoreColorClass(score)}">${score}/100</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
                ` : ''}

                <div class="row">
                    <div class="col-md-6">
                        ${techniqueAnalysis.strengths && techniqueAnalysis.strengths.length > 0 ? `
                        <div class="analysis-section">
                            <h6>Strengths</h6>
                            <ul class="list-unstyled">
                                ${techniqueAnalysis.strengths.map(strength => `
                                    <li>✅ ${strength.category?.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())} - ${Math.round(strength.score || 0)}/100</li>
                                `).join('')}
                            </ul>
                        </div>
                        ` : ''}
                    </div>
                    <div class="col-md-6">
                        ${techniqueAnalysis.weaknesses && techniqueAnalysis.weaknesses.length > 0 ? `
                        <div class="analysis-section">
                            <h6>Areas for Improvement</h6>
                            <ul class="list-unstyled">
                                ${techniqueAnalysis.weaknesses.map(weakness => `
                                    <li>⚠️ ${weakness.category?.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())} - ${Math.round(weakness.score || 0)}/100</li>
                                `).join('')}
                            </ul>
                        </div>
                        ` : ''}
                    </div>
                </div>

                ${improvementPlan.recommendations && improvementPlan.recommendations.length > 0 ? `
                <div class="analysis-section">
                    <h6>Recommendations</h6>
                    <ul class="recommendation-list">
                        ${improvementPlan.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                    </ul>
                </div>
                ` : ''}

                ${improvementPlan.practice_drills && improvementPlan.practice_drills.length > 0 ? `
                <div class="analysis-section">
                    <h6>Practice Drills</h6>
                    <ul class="drill-list">
                        ${improvementPlan.practice_drills.map(drill => `<li>${drill}</li>`).join('')}
                    </ul>
                </div>
                ` : ''}
            </div>
        `;
    }

    getScoreColorClass(score) {
        if (score >= 85) return 'text-success';
        if (score >= 75) return 'text-info';
        if (score >= 65) return 'text-warning';
        return 'text-danger';
    }

    async loadUploadedVideos() {
        try {
            const response = await fetch('/api/videos');
            const data = await response.json();
            
            if (response.ok && data.videos) {
                this.uploadedVideos = data.videos;
                this.displayUploadedVideos(data.videos);
            }
        } catch (error) {
            console.error('Failed to load videos:', error);
        }
    }

    displayUploadedVideos(videos) {
        const videosList = document.getElementById('videosList');
        if (!videosList) return;

        if (videos.length === 0) {
            videosList.innerHTML = '<p class="text-muted">No videos uploaded yet.</p>';
            return;
        }

        const videosHTML = videos.map(video => `
            <div class="video-item p-3 mb-3 border rounded">
                <div class="row align-items-center">
                    <div class="col-md-6">
                        <h6>${video.filename}</h6>
                        <small class="text-muted">
                            Uploaded: ${new Date(video.upload_time).toLocaleString()}<br>
                            Size: ${(video.size / (1024 * 1024)).toFixed(2)} MB
                        </small>
                    </div>
                    <div class="col-md-6">
                        <div class="d-flex gap-2">
                            <button class="btn btn-primary btn-sm analyze-btn" 
                                    data-file-id="${video.file_id}" 
                                    data-analysis-type="bowling">
                                ${video.bowling_analyzed ? '✓' : ''} Bowling
                            </button>
                            <button class="btn btn-success btn-sm analyze-btn" 
                                    data-file-id="${video.file_id}" 
                                    data-analysis-type="batting">
                                ${video.batting_analyzed ? '✓' : ''} Batting
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');

        videosList.innerHTML = videosHTML;
    }

    showProgress(message) {
        const progressContainer = document.getElementById('progressContainer');
        const progressMessage = document.getElementById('progressMessage');
        
        if (progressContainer && progressMessage) {
            progressMessage.textContent = message;
            progressContainer.style.display = 'block';
        }
    }

    hideProgress() {
        const progressContainer = document.getElementById('progressContainer');
        if (progressContainer) {
            progressContainer.style.display = 'none';
        }
    }

    showAlert(message, type = 'info') {
        const alertsContainer = document.getElementById('alertsContainer');
        if (!alertsContainer) return;

        const alertId = 'alert-' + Date.now();
        const alertHTML = `
            <div id="${alertId}" class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;

        alertsContainer.insertAdjacentHTML('beforeend', alertHTML);

        // Auto-remove alert after 5 seconds
        setTimeout(() => {
            const alertElement = document.getElementById(alertId);
            if (alertElement) {
                alertElement.remove();
            }
        }, 5000);
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new CricketAnalysisApp();
});

use std::path::PathBuf;

/// Single media item discovered by the unified scanner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaItem {
    Video(PathBuf),
    Image(PathBuf),
}

/// Outcome of processing a single media file.
#[derive(Debug)]
pub enum ProcessResult {
    Converted(FileStats),
    Skipped(String),
}

/// Size and encoding stats for one converted file.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FileStats {
    pub input_size: u64,
    pub output_size: u64,
    #[allow(dead_code)]
    pub used_crf: u8,
}

/// Aggregate statistics for a single media category (video or image).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MediaStats {
    pub successful: usize,
    pub skipped: usize,
    pub failed: usize,
    pub total_input_size: u64,
    pub total_output_size: u64,
}

impl MediaStats {
    pub fn record_converted(&mut self, stats: FileStats) {
        self.successful += 1;
        self.total_input_size += stats.input_size;
        self.total_output_size += stats.output_size;
    }

    pub fn record_skipped(&mut self) {
        self.skipped += 1;
    }

    pub fn record_failed(&mut self) {
        self.failed += 1;
    }

    pub fn total_processed(&self) -> usize {
        self.successful + self.skipped + self.failed
    }

    pub fn savings_percent(&self) -> f64 {
        if self.total_input_size == 0 {
            0.0
        } else {
            (1.0 - self.total_output_size as f64 / self.total_input_size as f64) * 100.0
        }
    }
}

/// Combined statistics across videos and images.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ConversionStats {
    pub video: MediaStats,
    pub image: MediaStats,
}

impl ConversionStats {
    pub fn combined(&self) -> MediaStats {
        MediaStats {
            successful: self.video.successful + self.image.successful,
            skipped: self.video.skipped + self.image.skipped,
            failed: self.video.failed + self.image.failed,
            total_input_size: self.video.total_input_size + self.image.total_input_size,
            total_output_size: self.video.total_output_size + self.image.total_output_size,
        }
    }
}

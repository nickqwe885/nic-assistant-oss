use thiserror::Error;

#[allow(dead_code)]
#[derive(Debug, Error)]
pub enum NisError {
    #[error("embed error: {0}")]
    Embed(String),

    #[error("llm inference: {0}")]
    Llm(String),

    #[error("database: {0}")]
    Database(String),

    #[error("encryption: {0}")]
    Crypto(String),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("config: {0}")]
    Config(String),

    #[error("task join: {0}")]
    Join(#[from] tokio::task::JoinError),

    #[error("serialization: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

#[allow(dead_code)]
pub type NisResult<T> = Result<T, NisError>;

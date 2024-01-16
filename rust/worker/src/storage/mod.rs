use async_trait::async_trait;

#[async_trait]
trait Storage {
    async fn get(&self, key: &str, path: &str) -> Result<(), String>;
    async fn put(&self, key: &str, path: &str) -> Result<(), String>;
}

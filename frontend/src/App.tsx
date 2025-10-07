import { useEffect, useState } from "react";

function App() {
  const [status, setStatus] = useState("loading...");

  useEffect(() => {
    fetch("http://localhost:8000/health")
      .then(r => r.json())
      .then(d => setStatus(d.status))
      .catch(() => setStatus("error"));
  }, []);

  const onUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.[0]) return;
    const form = new FormData();
    form.append("file", e.target.files[0]);
    const res = await fetch("http://localhost:8000/estimate", {
      method: "POST",
      body: form
    });
    const data = await res.json();
    alert(JSON.stringify(data, null, 2));
  };

  return (
    <div style={{ padding: 24 }}>
      <h1>AI Meal Calorie Estimator</h1>
      <p>Backend health: {status}</p>
      <input type="file" accept="image/*" onChange={onUpload} />
    </div>
  );
}

export default App;

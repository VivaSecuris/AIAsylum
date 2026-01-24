import { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

export default function Home() {
  const [testRuns, setTestRuns] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchTestRuns();
  }, []);

  const fetchTestRuns = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/v1/test-runs/`);
      setTestRuns(response.data);
    } catch (error) {
      console.error('Error fetching test runs:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '2rem', fontFamily: 'system-ui' }}>
      <h1>AI Asylum Dashboard</h1>
      <p>LLM Psychoanalysis Framework</p>
      
      <div style={{ marginTop: '2rem' }}>
        <h2>Test Runs</h2>
        {loading ? (
          <p>Loading...</p>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{ border: '1px solid #ddd', padding: '8px' }}>ID</th>
                <th style={{ border: '1px solid #ddd', padding: '8px' }}>Type</th>
                <th style={{ border: '1px solid #ddd', padding: '8px' }}>Status</th>
                <th style={{ border: '1px solid #ddd', padding: '8px' }}>Doctor</th>
                <th style={{ border: '1px solid #ddd', padding: '8px' }}>Patient</th>
              </tr>
            </thead>
            <tbody>
              {testRuns.map((run: any) => (
                <tr key={run.id}>
                  <td style={{ border: '1px solid #ddd', padding: '8px' }}>{run.id}</td>
                  <td style={{ border: '1px solid #ddd', padding: '8px' }}>{run.test_type}</td>
                  <td style={{ border: '1px solid #ddd', padding: '8px' }}>{run.status}</td>
                  <td style={{ border: '1px solid #ddd', padding: '8px' }}>{run.doctor_model}</td>
                  <td style={{ border: '1px solid #ddd', padding: '8px' }}>{run.patient_model}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

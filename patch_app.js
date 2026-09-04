const fs = require('fs');

let appJs = fs.readFileSync('frontend/App.js', 'utf8');

// 1. Add import and inferenceMode state
appJs = appJs.replace("import { useState, useEffect, useRef } from 'react';", "import { useState, useEffect, useRef } from 'react';\nimport { AILocalEngine } from './ai_inference';\nimport { Switch } from 'react-native';");
appJs = appJs.replace("const [connected, setConnected] = useState(false);", "const [connected, setConnected] = useState(false);\n  const [inferenceMode, setInferenceMode] = useState('cloud');\n  const aiEngine = useRef(null);");

// 2. Initialize aiEngine
appJs = appJs.replace("useEffect(() => {", "useEffect(() => {\n    if (!aiEngine.current) {\n      aiEngine.current = new AILocalEngine();\n    }");

// 3. Process telemetry to override velocity in device mode
const messageHandlerOld = `
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        setTelemetry(data);
`;
const messageHandlerNew = `
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        
        // On-Device Edge Inference Override
        if (inferenceMode === 'device' && data.measured.imu) {
            const localSpeed = aiEngine.current.predict(data.measured.imu);
            data.estimated.velocity = localSpeed;
            data.estimated.mode = "On-Device Local Inference";
        }
        
        setTelemetry(data);
`;
appJs = appJs.replace(messageHandlerOld, messageHandlerNew);

// 4. Add UI toggle
const toggleUI = `
        <Text style={styles.statusText}>Connection: {connected ? '🟢 ONLINE' : '🔴 OFFLINE'}</Text>
        
        <View style={{flexDirection: 'row', alignItems: 'center', marginTop: 10}}>
          <Text style={{color: 'white', marginRight: 10, fontWeight: 'bold'}}>
             {inferenceMode === 'device' ? '🧠 On-Device Edge AI' : '☁️ Cloud / Server AI'}
          </Text>
          <Switch 
            value={inferenceMode === 'device'} 
            onValueChange={(val) => setInferenceMode(val ? 'device' : 'cloud')}
            trackColor={{ false: "#767577", true: "#81b0ff" }}
            thumbColor={inferenceMode === 'device' ? "#2196F3" : "#f4f3f4"}
          />
        </View>
`;
appJs = appJs.replace("<Text style={styles.statusText}>Connection: {connected ? '🟢 ONLINE' : '🔴 OFFLINE'}</Text>", toggleUI);

fs.writeFileSync('frontend/App.js', appJs);

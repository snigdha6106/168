const fs = require('fs');
let appJs = fs.readFileSync('frontend/App.js', 'utf8');

// Replace the inferenceMode state with a ref-backed approach or just use a ref directly for the callback
appJs = appJs.replace("const [inferenceMode, setInferenceMode] = useState('cloud');", "const [inferenceMode, setInferenceMode] = useState('cloud');\n  const modeRef = useRef('cloud');");

// Update the toggle to also update the ref
appJs = appJs.replace("onValueChange={(val) => setInferenceMode(val ? 'device' : 'cloud')}", "onValueChange={(val) => { const newMode = val ? 'device' : 'cloud'; setInferenceMode(newMode); modeRef.current = newMode; }}");

// Update the message handler to read from modeRef.current
appJs = appJs.replace("if (inferenceMode === 'device' && data.measured.imu)", "if (modeRef.current === 'device' && data.measured.imu)");

fs.writeFileSync('frontend/App.js', appJs);

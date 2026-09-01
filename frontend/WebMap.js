import React, { useEffect } from 'react';
import { MapContainer, TileLayer, Polyline, Marker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

const carIcon = L.divIcon({
  html: '<div style="width: 20px; height: 20px; background-color: #FFEB3B; border-radius: 50%; border: 2px solid black; box-shadow: 0 0 5px rgba(0,0,0,0.5);"></div>',
  className: '',
  iconSize: [20, 20],
  iconAnchor: [10, 10]
});

function MapUpdater({ center }) {
  const map = useMap();
  useEffect(() => {
    if (center && center[0] !== 0) {
      map.setView(center, map.getZoom());
    }
  }, [center, map]);
  return null;
}

export default function WebMap({ path, truthPath, currentPos }) {
  const center = currentPos ? [currentPos.latitude, currentPos.longitude] : [28.6139, 77.2090];
  
  // Convert {latitude, longitude} to [lat, lon] for Leaflet
  const formattedPath = path.map(p => [p.latitude, p.longitude]);
  const formattedTruth = truthPath.map(p => [p.latitude, p.longitude]);

  return (
    <div style={{ flex: 1, width: '100%', height: '100%' }}>
      <MapContainer 
        center={center} 
        zoom={17} 
        style={{ height: '100%', width: '100%' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        
        {formattedPath.length > 0 && (
          <Polyline positions={formattedPath} color="#2196F3" weight={5} />
        )}
        
        {formattedTruth.length > 0 && (
          <Polyline positions={formattedTruth} color="#4CAF50" weight={3} dashArray="5, 10" />
        )}

        {currentPos && (
          <Marker position={center} icon={carIcon}>
            <Popup>Vehicle is here</Popup>
          </Marker>
        )}
        
        <MapUpdater center={center} />
      </MapContainer>
    </div>
  );
}

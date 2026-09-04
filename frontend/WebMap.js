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

function MapUpdater({ center, isFollowing, onInteraction }) {
  const map = useMap();
  useEffect(() => {
    if (center && center[0] !== 0 && isFollowing) {
      map.setView(center, map.getZoom(), { animate: true });
    }
  }, [center, map, isFollowing]);

  useEffect(() => {
    if (!onInteraction) return;
    map.on('dragstart', onInteraction);
    map.on('zoomstart', onInteraction);
    return () => {
      map.off('dragstart', onInteraction);
      map.off('zoomstart', onInteraction);
    }
  }, [map, onInteraction]);
  
  return null;
}

export default function WebMap({ path, truthPath, currentPos, isFollowing, onInteraction }) {
  const center = currentPos ? [currentPos.latitude, currentPos.longitude] : [28.6139, 77.2090];
  
  const formattedPath = path.map(p => [p.latitude, p.longitude]);
  const formattedTruth = truthPath.map(p => [p.latitude, p.longitude]);

  return (
    <div style={{ flex: 1, width: '100%', height: '100%' }}>
      <MapContainer center={center} zoom={17} style={{ height: '100%', width: '100%' }}>
        <TileLayer attribution='&copy; OSM' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        {formattedPath.length > 0 && <Polyline positions={formattedPath} color="#2196F3" weight={5} />}
        {formattedTruth.length > 0 && <Polyline positions={formattedTruth} color="#4CAF50" weight={3} dashArray="5, 10" />}
        {currentPos && <Marker position={center} icon={carIcon}><Popup>Vehicle is here</Popup></Marker>}
        <MapUpdater center={center} isFollowing={isFollowing} onInteraction={onInteraction} />
      </MapContainer>
    </div>
  );
}

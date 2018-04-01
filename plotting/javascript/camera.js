var isMouseDown = false;
var onMouseDownPosition = new THREE.Vector2();

var theta = 0;
var onMouseDownTheta = 0;
var phi = 0;
var onMouseDownPhi = 0;



function onDocumentMouseWheel( event ) {

	//			Standard				IE						Firefox
    camRad += -10*(event.wheelDeltaY || event.wheelDelta) || 1000*event.detail;
	if(camRad < 100 )
	{
		camRad = 100;
	}

	camUpdate();

}



function onDocumentMouseDown( event ) {

	event.preventDefault();

	isMouseDown = true;

	onMouseDownTheta = theta;
	onMouseDownPhi = phi;
	onMouseDownPosition.x = event.clientX;
	onMouseDownPosition.y = event.clientY;

}
			
function onDocumentMouseMove( event ) {

	event.preventDefault();

	if ( isMouseDown ) {

		theta = - ( ( event.clientX - onMouseDownPosition.x ) * 0.5 ) + onMouseDownTheta;
		phi = ( ( event.clientY - onMouseDownPosition.y ) * 0.5 ) + onMouseDownPhi;

		phi = Math.min( 180, Math.max( -180, phi ) );

		camUpdate();

	}

}
			
function onDocumentMouseUp( event ) {

	event.preventDefault();

	isMouseDown = false;

}

function camUpdate(){
	
	camera.position.x = camRad * Math.cos( theta * Math.PI / 360 ) * Math.cos( phi * Math.PI / 360 );
	camera.position.y = camRad * Math.sin( theta * Math.PI / 360 ) * Math.cos( phi * Math.PI / 360 );
	camera.position.z = camRad * Math.sin( phi * Math.PI / 360 );
	
	camera.lookAt(new THREE.Vector3( 0, 0, 0 ));
	camera.updateMatrix();
	
}
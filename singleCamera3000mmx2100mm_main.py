

from singleCamera3000mmx2100mm_module import *

async def handle_client(websocket, path):
    print("✅ Unreal 클라이언트가 연결되었습니다.")
    global left_src_points, right_src_points, left_dst_points, right_dst_points, left_previous_static_hand_events, right_previous_static_hand_events
    left_pipeline, left_align, right_pipeline, right_align = init_camera()
    try:
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=2)
        Left_Color, Left_Depth, Left_Raw_Depth_Frame= await loop.run_in_executor(executor, Get_Frame, left_pipeline, left_align)
        Right_Color, Right_Depth, Right_Raw_Depth_Frame = await loop.run_in_executor(executor, Get_Frame, right_pipeline, right_align)

        
        Left_Zwall = Get_Zwall(Left_Depth, left_src_points, left_pipeline, left_align)#왼쪽 Z Wall 
        Right_Zwall = Get_Zwall(Right_Depth, right_src_points, right_pipeline, right_align)#오른쪽 Z Wall
        start = time.time()
        count = 0
        prev = []
        while True:
            end = time.time()
            count += 1
            if end-start > 1:
                start = time.time()
                print(f"현재프레임 : {count}")
                count = 0
            future_left = loop.run_in_executor(executor, all_process_for_detect, left_pipeline, left_align, left_src_points, left_dst_points, Left_Zwall, left_previous_static_hand_events, "left")
            future_right = loop.run_in_executor(executor, all_process_for_detect, right_pipeline, right_align, right_src_points, right_dst_points, Right_Zwall, right_previous_static_hand_events, "right")

            # 두 작업이 모두 끝날 때까지 기다림 (병렬 실행 + 동시 대기)
            (left_RealFinal_data, img1, deletedList1), (right_RealFinal_data, img2, deletedList2) = await asyncio.gather(future_left, future_right)
            dst = np.float32([[0,0],[1280,0],[1280,720],[0,720]])
            matrix1 = cv2.getPerspectiveTransform(left_src_points, dst)
            #matrix2 = cv2.getPerspectiveTransform(right_src_points, dst)
            # 투시 변환 적용
            warped1 = cv2.warpPerspective(img1, matrix1, (1280, 720))
            #warped2 = cv2.warpPerspective(img2, matrix2, (1280, 720))
            #warped2=warped2[:,120:1280]
            #horizontal = np.hstack((warped1, warped2))
            batch_data = [] # left_batch_data와 right_batch_data 처리 후 실행
            #Realfinal_unique, armUnique =merge_and_deduplicate(horizontal,left_RealFinal_data, right_RealFinal_data, deletedList1, deletedList2)
            Realfinal_unique, armUnique, deletedListMerged =merge_and_deduplicate(warped1,left_RealFinal_data, right_RealFinal_data, deletedList1, deletedList2, left_previous_static_hand_events, right_previous_static_hand_events)
            # for lidar_data in final_Lidar_events:
            #     data_dict = {
            #         "IsAttached": "true",
            #         "object_type": "hand",
            #         "Relative_x": float(lidar_data.wallx/wall_width),
            #         "Relative_y": float(lidar_data.wally/wall_height),
            #         "timestamp": time.time()
            #     }
            #     batch_data.append(data_dict)

            print("-"*50)
            for start, arm in enumerate(armUnique):
                print(f"arm data{start}: {arm}")
            print(f"arm data count : {FinalArmData.count}")
            print("-"*50)

            FinalArmData.count = 0

            
            
            for Data in Realfinal_unique:
                data_dict = {
                    "IsAttached": Data.IsAttached,
                    "object_type": Data.label,
                    "Relative_x": float(Data.relative_x),
                    "Relative_y": float(Data.relative_y),
                    #"timestamp": time.time()
                }

                batch_data.append(data_dict)
            for Data in deletedListMerged:
                data_dict = {
                    "IsAttached": False,
                    "object_type": Data.label,
                    "Relative_x": float(Data.relative_x),
                    "Relative_y": float(Data.relative_y),
                    #"timestamp": time.time()
                }
            for arm in armUnique:
                data_dict = {
                    "IsAttached": False,
                    "object_type": arm.label,
                    "Relative_x": float((arm.x1+arm.x2)/2),
                    "Relative_y": float((arm.y1 + arm.y2)/2),
                    #"timestamp": time.time()
                }
            
            if batch_data:
                print("📤 데이터 전송 이전")
                await websocket.send(json.dumps(batch_data))
                #asyncio.create_task(websocket.send(orjson.dumps(batch_data)))
                print("📤 데이터 전송 완료")

            # 🔻 q 누르면 루프 종료
            if keyboard.is_pressed('q'):
                print("🟥 q 키 입력 감지됨. 서버 종료합니다.")
                break

            await asyncio.sleep(1 /Per_Frame)  # 프레임 속도 제한


    finally:
        executor.shutdown(wait=True)
        left_pipeline.stop()
        right_pipeline.stop()
        print("🛑 카메라 종료 및 리소스 해제 완료")

    
        
async def main():
    print("[INFO] WebSocket 서버 시작 중...")
    '''async with websockets.serve(handle_client, "0.0.0.0", 8765):
        await asyncio.Future()  # run forever'''
    #await handle_client(None, None)
if __name__ == "__main__":
    asyncio.run(main())

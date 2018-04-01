#ifndef GFC_GNSSCONST_H
#define GFC_GNSSCONST_H

//============================================================================
//
//  This file is part of GFC, the GNSS FOUNDATION CLASS.
//
//  The GFC is free software; you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License as published
//  by the Free Software Foundation; either version 3.0 of the License, or
//  any later version.
//
//  The GFC is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
//
//  You should have received a copy of the GNU Lesser General Public
//  License along with GFC; if not, write to the Free Software Foundation,
//  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110, USA
//
//  Copyright 2015, lizhen
//
//============================================================================


/*
 
 GNSS���ݴ�������еĳ����Ĵ洢
 
 ���еĳ�Ա����ȫ������Ϊstatic���������
 
 �����еĳ���ȫ���ڳ�ʼ��ʱ���ļ��ж�ȡ��
 δ���滻�������г����Ķ����ļ�constant.h
 
 ���������� 2015��2��3��
 
 */


/*
 ÿ�����ǵ���άģ�Ͷ���ṹ��
 Ŀǰ�����뷨�ǿ�����DTN����������������άģ��
 �����������ı�����ϣ��������������ڣ�Ԥ��������
 ÿ�ŷ������������ͺţ��Լ����屾������ϵ�Ķ���
 ��������ʵʱ��̬
 
 */

#include "GString.h"
#include <map>
#include <iomanip>
#include <math.h>

namespace gfc
{
    
    /*����ο�ϵͳ�Ķ���
     WGS-84����ϵ��G1150�ܵĵ�һ����ʱ
     CGCS200����ϵ�Ĳο���Ԫ�ǣ�����
     ITRS��Z��ָ��ΪBIH1984.0
     ����ITRS���ú��ֲο�����????
     
     */
    //	struct RefSys
    //	{
    //		GString  name;  //����ϵ������
    //		Ellipsoid ellipsoid;  //ÿ������ϵ����һ���ο�����
    //		double T0;    // ÿ������ϵ����һ���ο���ʼ��Ԫʱ��(MJD)������ȷ��������ָ��
    //	};
    
    
    ///*GNSS����ϵͳ�Ķ���*/
    //struct GNSSSYS
    //{
    //	int sys;    //ϵͳ��ǣ�Ŀǰ��GPSΪ0x01,BDSΪ0x02,GLONASSΪ0x04, GALLILUEΪ0x08
    //	std::string sysname;  //����ϵͳ������
    //	char signalType;  //�ź����ͣ�0ΪCDMA��1ΪFDMA��2ΪTDMA��Ŀǰֻ��GLONASSΪFDMA���Ǽ���·���ݲ���TDMA��ʽ
    //	int satnum;  //������ϵͳ�����ǿ���
    //	double T0;   //������ϵͳʱ�����ʼʱ��(MJD��ʾ)
    //	RefSys cs;  //������ϵͳ���õ�����ϵͳ
    //	std::vector<SpaceCraftModel>  spaceCraft;  //������ϵͳ�����õ����еķ�������Ϣ
    //};
    
    
    NEW_GEXCEPTION_CLASS( constantUnexist, gfc::GException );
    
    class GFCCONST
    {
        
    public:
        
        /*
         ��̬���������ڶ�constantValue���г�ʼ��
         */
        static std::map< GString,  long double > Initializer();
        static void RegByName(GString variableName,long double variableValue);
        static long double GetByName(GString variableName);
        
        static void UnregByName(GString variableName);
        static void dump( std::ostream& s ) ;
        
    private:
        
        // private construction function means this class can not be hesitated!
       	GFCCONST(void);
        virtual ~GFCCONST(void);
        
        static long double GetByName_internal( GString variableName,std::map< GString,  long double >& myconstantValue);
        
        static std::map< GString,  long double > constantValue;  //���б�����ֵ�Լ����������洢������
        
        //GString constFileName;  //���������ļ�(���Ϊ�ı��ļ�)
        
        //std::map<Ellipsoid>   myEllipsoid;                        //���е�������Ϣ
        
    };
    
    
    
    
    
    
}  // end of namespace


/*
 
 �����CONSTANT���ڻ�ȡ������ֵ
 
 ���Է���LogStream����б�д
 
 ע�⣺��������Ϊ��д;
 
 ���ø�ʽ���£�
 CONSTANT("CLIHT")
 CONSTANT("PI")
 
 */

#define GCONST(constantName) \
GFCCONST::GetByName(constantName)

#endif


